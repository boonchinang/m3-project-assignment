import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="TSLA ML Trading Model Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E82127;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #E82127;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Header
st.header('⚡ TSLA ML Trading Model Dashboard')
st.markdown('<p class="sub-header">Algorithmic Trading, Feature Engineering, Custom Target Horizon & Thresholds, Machine Learning Signals & Backtesting</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.header("🛠️ Model & Strategy Controls")

ticker = st.sidebar.text_input("Stock Ticker", value="TSLA").upper()

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
with col_date2:
    end_date = st.date_input("End Date", value=datetime.date.today())

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Target Definition")
st.sidebar.caption("Define the future return look-ahead horizon & buy/sell classification thresholds.")

future_period = st.sidebar.slider(
    "Look-Ahead Horizon (Trading Days)",
    min_value=1,
    max_value=60,
    value=30,
    step=1,
    help="FUTURE_PERIOD: Number of trading days into the future to measure returns."
)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    buy_threshold_pct = st.number_input(
        "Buy Thresh (%)",
        min_value=0.0,
        max_value=50.0,
        value=6.0,
        step=0.5,
        help="BUY_THRESHOLD: Future return > +X% triggers Buy (1) signal."
    )
with col_t2:
    sell_threshold_pct = st.number_input(
        "Sell Thresh (%)",
        min_value=-50.0,
        max_value=0.0,
        value=-6.0,
        step=0.5,
        help="SELL_THRESHOLD: Future return < -X% triggers Sell (-1) signal."
    )

buy_threshold = buy_threshold_pct / 100.0
sell_threshold = sell_threshold_pct / 100.0

allow_shorting = st.sidebar.checkbox("Allow Short Selling (-1 Position)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Technical Indicators")
sma_short_win = st.sidebar.slider("Short SMA Window", 5, 50, 20)
sma_long_win = st.sidebar.slider("Long SMA Window", 20, 200, 50)
rsi_win = st.sidebar.slider("RSI Period", 5, 30, 14)
bb_win = st.sidebar.slider("Bollinger Bands Window", 10, 50, 20)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Machine Learning Model")
model_type = st.sidebar.selectbox(
    "Select Classifier",
    ["Random Forest", "Gradient Boosting", "Logistic Regression", "Support Vector Machine"]
)
test_size = st.sidebar.slider("Test Set Ratio", 0.1, 0.4, 0.2, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Backtest Settings")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=1000)
commission = st.sidebar.number_input("Transaction Fee (%)", value=0.1, step=0.05) / 100.0

# ---------------------------------------------------------
# Helper Functions: Data Fetching & Feature Engineering
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start, end):
    """
    Generate or fetch stock data for demo/interactive analysis.
    Uses yfinance if available, or realistic simulation fallback if offline.
    """
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        if not df.empty:
            return df
    except Exception:
        pass

    # Synthetic realistic stock price generator fallback
    dates = pd.date_range(start=start, end=end, freq='B')
    np.random.seed(42)
    returns = np.random.normal(0.0012, 0.035, len(dates)) # TSLA-like volatility
    price = 150.0 * np.exp(np.cumsum(returns))
    
    high = price * (1 + np.abs(np.random.normal(0, 0.015, len(dates))))
    low = price * (1 - np.abs(np.random.normal(0, 0.015, len(dates))))
    open_p = price * (1 + np.random.normal(0, 0.008, len(dates)))
    volume = np.random.randint(20000000, 100000000, len(dates))

    df = pd.DataFrame({
        'Open': open_p,
        'High': high,
        'Low': low,
        'Close': price,
        'Adj Close': price,
        'Volume': volume
    }, index=dates)
    return df

def engineer_features(df, sma_short, sma_long, rsi_p, bb_p, horizon, buy_thresh, sell_thresh):
    data = df.copy()
    
    # Returns
    data['Return'] = data['Close'].pct_change()
    
    # Moving Averages
    data['SMA_Short'] = data['Close'].rolling(window=sma_short).mean()
    data['SMA_Long'] = data['Close'].rolling(window=sma_long).mean()
    data['SMA_Ratio'] = data['SMA_Short'] / data['SMA_Long']
    
    # RSI Calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_p).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_p).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=bb_p).mean()
    data['BB_Std'] = data['Close'].rolling(window=bb_p).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volatility & Momentum
    data['Volatility_10D'] = data['Return'].rolling(window=10).std()
    data['Mom_5D'] = data['Close'].pct_change(5)
    
    # Future Look-ahead Return Target Definition
    data['Future_Return'] = (data['Close'].shift(-horizon) - data['Close']) / data['Close']
    
    # Multiclass Classification:
    # 1  : Buy  (Future Return > buy_thresh)
    # -1 : Sell (Future Return < sell_thresh)
    # 0  : Hold / Neutral (In between thresholds)
    conditions = [
        data['Future_Return'] > buy_thresh,
        data['Future_Return'] < sell_thresh
    ]
    choices = [1, -1]
    data['Target'] = np.select(conditions, choices, default=0)
    
    data.dropna(inplace=True)
    return data

# ---------------------------------------------------------
# Load & Process Data
# ---------------------------------------------------------
raw_df = load_stock_data(ticker, start_date, end_date)

if raw_df.empty:
    st.error("No data found for the given symbol and date range.")
    st.stop()

df_feat = engineer_features(
    raw_df, sma_short_win, sma_long_win, rsi_win, bb_win,
    future_period, buy_threshold, sell_threshold
)

# Feature Selection
feature_cols = ['Return', 'SMA_Ratio', 'RSI', 'BB_Width', 'MACD', 'MACD_Signal', 'Volatility_10D', 'Mom_5D']
X = df_feat[feature_cols]
y = df_feat['Target']

# Time Series / Sequential Split
split_idx = int(len(df_feat) * (1 - test_size))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
df_test = df_feat.iloc[split_idx:].copy()

# Model Initialization
if model_type == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
elif model_type == "Gradient Boosting":
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
elif model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = SVC(probability=True, kernel='rbf', C=1.0)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
df_test['Signal'] = y_pred

# Generate Position
if allow_shorting:
    df_test['Position'] = df_test['Signal'] # 1, 0, or -1
else:
    df_test['Position'] = np.where(df_test['Signal'] == 1, 1, 0) # Only Long or Cash

df_test['Strategy_Return'] = df_test['Position'].shift(1) * df_test['Return'] - (np.abs(df_test['Position'].diff()) * commission)
df_test.dropna(inplace=True)

# Cumulative Returns
df_test['Cum_Benchmark'] = (1 + df_test['Return']).cumprod() * initial_capital
df_test['Cum_Strategy'] = (1 + df_test['Strategy_Return']).cumprod() * initial_capital

# ---------------------------------------------------------
# Top Summary Metrics & Target Banner
# ---------------------------------------------------------
latest_close = float(raw_df['Close'].iloc[-1])
prev_close = float(raw_df['Close'].iloc[-2])
price_change = latest_close - prev_close
pct_change = (price_change / prev_close) * 100

acc = accuracy_score(y_test.loc[df_test.index], y_pred[:len(df_test)])
strat_total_return = ((df_test['Cum_Strategy'].iloc[-1] - initial_capital) / initial_capital) * 100
bench_total_return = ((df_test['Cum_Benchmark'].iloc[-1] - initial_capital) / initial_capital) * 100

# Target Parameter Status Card
st.info(
    f"🎯 **Target Configuration Active**: Look-Ahead Horizon = **{future_period} Trading Days** | "
    f"Buy Threshold (`1`) = **> +{buy_threshold_pct:.1f}%** | "
    f"Sell Threshold (`-1`) = **< {sell_threshold_pct:.1f}%** | "
    f"Neutral (`0`) = **Between thresholds**"
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Latest TSLA Close", f"${latest_close:.2f}", f"{pct_change:+.2f}%")
m2.metric("ML Model Accuracy", f"{acc*100:.1f}%")
m3.metric("Strategy Return", f"{strat_total_return:+.1f}%")
m4.metric("Buy & Hold Return", f"{bench_total_return:+.1f}%")
m5.metric("Final Portfolio", f"${df_test['Cum_Strategy'].iloc[-1]:,.2f}")

st.markdown("---")

# ---------------------------------------------------------
# Tabs Section
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price & Indicators", 
    "🤖 ML Model Analysis", 
    "📈 Strategy Backtest", 
    "🔮 Live Signal Sandbox"
])

# ---------------------------------------------------------
# TAB 1: Technical Analysis
# ---------------------------------------------------------
with tab1:
    st.subheader("Price History & Technical Overlays")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=('Price & Moving Averages', 'Relative Strength Index (RSI)', 'MACD'),
                        row_heights=[0.5, 0.25, 0.25])
    
    # Candlestick / Line
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df['Close'], name='Close Price', line=dict(color='#E82127', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_Short'], name=f'SMA {sma_short_win}', line=dict(color='#0066CC', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_Long'], name=f'SMA {sma_long_win}', line=dict(color='#FF9900', width=1)), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['RSI'], name='RSI', line=dict(color='#8884d8')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['MACD_Signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    
    fig.update_layout(height=700, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: ML Model Metrics
# ---------------------------------------------------------
with tab2:
    st.subheader(f"Model Evaluation: {model_type}")
    
    c1, c2 = st.columns(2)
    
    y_eval = y_test.loc[df_test.index]
    p_eval = y_pred[:len(df_test)]
    unique_classes = sorted(list(set(y_eval).union(set(p_eval))))
    class_labels = [f"Sell (-1)" if c == -1 else f"Neutral (0)" if c == 0 else f"Buy (1)" for c in unique_classes]

    with c1:
        st.write("#### Confusion Matrix")
        cm = confusion_matrix(y_eval, p_eval, labels=unique_classes)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=class_labels,
            y=class_labels,
            color_continuous_scale='Reds'
        )
        fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with c2:
        st.write("#### Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
            fig_imp.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance plot is available for tree-based models (Random Forest, Gradient Boosting).")
            
    st.write("#### Detailed Classification Metrics (Weighted)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    prec = precision_score(y_eval, p_eval, average='weighted', zero_division=0)
    rec = recall_score(y_eval, p_eval, average='weighted', zero_division=0)
    f1 = f1_score(y_eval, p_eval, average='weighted', zero_division=0)
    
    m_col1.metric("Precision (Weighted)", f"{prec:.3f}")
    m_col2.metric("Recall (Weighted)", f"{rec:.3f}")
    m_col3.metric("F1 Score (Weighted)", f"{f1:.3f}")
    m_col4.metric("Test Samples", f"{len(y_eval)}")

# ---------------------------------------------------------
# TAB 3: Strategy Backtest
# ---------------------------------------------------------
with tab3:
    st.subheader("Algorithmic Trading Backtest Performance")
    
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(x=df_test.index, y=df_test['Cum_Strategy'], name='ML Strategy', line=dict(color='#00CC66', width=2.5)))
    fig_backtest.add_trace(go.Scatter(x=df_test.index, y=df_test['Cum_Benchmark'], name='Buy & Hold TSLA', line=dict(color='#E82127', width=1.5, dash='dash')))
    
    fig_backtest.update_layout(
        title="Portfolio Growth ($)",
        yaxis_title="Portfolio Value ($)",
        height=450,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_backtest, use_container_width=True)
    
    # Backtest Performance Metrics
    daily_strat_ret = df_test['Strategy_Return']
    sharpe = (daily_strat_ret.mean() / (daily_strat_ret.std() + 1e-9)) * np.sqrt(252)
    cum_max = df_test['Cum_Strategy'].cummax()
    drawdown = (df_test['Cum_Strategy'] - cum_max) / cum_max
    max_drawdown = drawdown.min() * 100
    win_rate = (daily_strat_ret > 0).mean() * 100
    
    st.write("#### Performance Summary")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Sharpe Ratio (Annualized)", f"{sharpe:.2f}")
    b2.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    b3.metric("Daily Win Rate", f"{win_rate:.1f}%")
    b4.metric("Total Trades Executed", f"{int(np.abs(df_test['Position'].diff()).sum())}")

# ---------------------------------------------------------
# TAB 4: Live Prediction Sandbox
# ---------------------------------------------------------
with tab4:
    st.subheader("🔮 Real-Time Trading Signal Predictor")
    st.write("Adjust indicator features below to test the trained model's target signal output in real-time:")
    
    s1, s2, s3 = st.columns(3)
    with s1:
        inp_return = st.slider("Daily Return", -0.10, 0.10, 0.01, 0.005)
        inp_sma_ratio = st.slider("SMA Ratio (Short/Long)", 0.80, 1.20, 1.02, 0.01)
        inp_rsi = st.slider("RSI (14)", 10.0, 90.0, 55.0, 1.0)
    with s2:
        inp_bb = st.slider("Bollinger Band Width", 0.01, 0.20, 0.05, 0.005)
        inp_macd = st.slider("MACD Value", -5.0, 5.0, 0.5, 0.1)
        inp_macd_sig = st.slider("MACD Signal Value", -5.0, 5.0, 0.2, 0.1)
    with s3:
        inp_vol = st.slider("10D Volatility", 0.005, 0.08, 0.025, 0.005)
        inp_mom = st.slider("5D Momentum", -0.15, 0.15, 0.02, 0.01)

    input_df = pd.DataFrame([{
        'Return': inp_return,
        'SMA_Ratio': inp_sma_ratio,
        'RSI': inp_rsi,
        'BB_Width': inp_bb,
        'MACD': inp_macd,
        'MACD_Signal': inp_macd_sig,
        'Volatility_10D': inp_vol,
        'Mom_5D': inp_mom
    }])

    pred_signal = model.predict(input_df)[0]
    
    st.markdown("---")
    st.write("### Predicted Target Signal:")
    if pred_signal == 1:
        st.success(f"🟢 **BUY SIGNAL (`1`)** — Expecting future return > +{buy_threshold_pct:.1f}% over the next {future_period} trading days.")
    elif pred_signal == -1:
        st.error(f"🔴 **SELL / SHORT SIGNAL (`-1`)** — Expecting future return < {sell_threshold_pct:.1f}% over the next {future_period} trading days.")
    else:
        st.warning(f"🟡 **NEUTRAL / HOLD SIGNAL (`0`)** — Expecting price movement between {sell_threshold_pct:.1f}% and +{buy_threshold_pct:.1f}%.")
