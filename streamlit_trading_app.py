import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="TSLA Trading Model & Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Header ---
st.title("📈 Tesla (TSLA) Algorithmic Trading Model Dashboard")
st.markdown("""
This interactive Streamlit application implements an ML-driven quantitative trading model for **Tesla, Inc. (TSLA)** and other tickers.
It features automated data fetching, technical feature engineering, Random Forest predictive modeling, signal generation, and backtesting performance comparison against Buy & Hold.
""")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Model & Data Parameters")

ticker = st.sidebar.text_input("Stock Ticker", value="TSLA").upper()

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
with col_date2:
    end_date = st.date_input("End Date", datetime.date.today())

# --- Sidebar Target Definition Controls ---
st.sidebar.subheader("🎯 Target Definition Controls")

future_period = st.sidebar.slider(
    "FUTURE_PERIOD (Look-Ahead Horizon in Days)", 
    min_value=1, max_value=60, value=20, step=1,
    help="Number of trading days ahead to evaluate future returns."
)

buy_threshold_pct = st.sidebar.number_input(
    "BUY_THRESHOLD (%)", 
    min_value=0.0, max_value=50.0, value=6.0, step=0.5,
    help="Future return > BUY_THRESHOLD triggers a Buy (+1) target signal."
)
buy_threshold = buy_threshold_pct / 100.0

sell_threshold_pct = st.sidebar.number_input(
    "SELL_THRESHOLD (%)", 
    min_value=-50.0, max_value=0.0, value=-6.0, step=0.5,
    help="Future return < SELL_THRESHOLD triggers a Sell (-1) target signal."
)
sell_threshold = sell_threshold_pct / 100.0

# --- 3-Class Target Labeling ---
def build_features_and_target(df, horizon, buy_th, sell_th):
    data = df.copy()
    data["Future_Return"] = data["Close"].shift(-horizon) / data["Close"] - 1.0
    data = data.dropna(subset=["Future_Return"])
    
    conditions = [
        data["Future_Return"] > buy_th,
        data["Future_Return"] < sell_th
    ]
    data["Target"] = np.select(conditions, [1, -1], default=0)
    return data

st.sidebar.subheader("📊 Technical Indicators")
short_window = st.sidebar.slider("Short SMA Window", 5, 50, 10)
long_window = st.sidebar.slider("Long SMA Window", 20, 200, 50)
rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14)

st.sidebar.subheader("🤖 Machine Learning Model")
n_estimators = st.sidebar.slider("Random Forest Trees", 50, 300, 100, step=25)
max_depth = st.sidebar.slider("Max Tree Depth", 3, 20, 8)
train_split = st.sidebar.slider("Train Data Ratio", 0.5, 0.9, 0.8, step=0.05)

st.sidebar.subheader("💰 Backtest Simulation")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, step=1000)
transaction_cost = st.sidebar.number_input("Transaction Cost per Trade (%)", value=0.1, step=0.05) / 100.0

# --- Data Fetching & Caching ---
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

df_raw = load_stock_data(ticker, start_date, end_date)

if df_raw is None or df_raw.empty:
    st.error("No stock data found for the selected ticker and date range. Please adjust sidebar parameters.")
    st.stop()

# --- Feature Engineering Function ---
def compute_features(df, s_win, l_win, rsi_p):
    data = df.copy()
    
    # Technical Indicators
    data['SMA_Short'] = data['Close'].rolling(window=s_win).mean()
    data['SMA_Long'] = data['Close'].rolling(window=l_win).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_p).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_p).mean()
    rs = gain / (loss + 1e-8)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])
    
    # Returns & Volatility
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
    
    # Lags for ML Features
    for i in range(1, 4):
        data[f'Return_Lag_{i}'] = data['Daily_Return'].shift(i)
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    
    # Target: 1 if next day's close > today's close, else 0
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    
    data.dropna(inplace=True)
    return data

df_feat = compute_features(df_raw, short_window, long_window, rsi_period)

feature_cols = [
    'SMA_Short', 'SMA_Long', 'MACD', 'MACD_Signal', 'RSI', 
    'BB_Upper', 'BB_Lower', 'Volatility', 
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3'
]

# --- Train ML Model ---
X = df_feat[feature_cols]
y = df_feat['Target']

split_idx = int(len(df_feat) * train_split)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

df_feat['Predicted_Signal'] = model.predict(X)
df_feat['Position'] = np.where(df_feat['Predicted_Signal'] == 1, 1, 0)

# Backtest Performance Calculation
df_feat['Strategy_Return'] = df_feat['Position'].shift(1) * df_feat['Daily_Return']
trades = df_feat['Position'].diff().abs().fillna(0)
df_feat['Strategy_Return'] -= trades * transaction_cost

df_feat['Cum_Market_Return'] = (1 + df_feat['Daily_Return']).cumprod()
df_feat['Cum_Strategy_Return'] = (1 + df_feat['Strategy_Return']).cumprod()

df_feat['Portfolio_Market'] = initial_capital * df_feat['Cum_Market_Return']
df_feat['Portfolio_Strategy'] = initial_capital * df_feat['Cum_Strategy_Return']

# Performance Metrics on Test Set
test_df = df_feat.iloc[split_idx:].copy()
accuracy = accuracy_score(y_test, model.predict(X_test))
precision = precision_score(y_test, model.predict(X_test), zero_division=0)
recall = recall_score(y_test, model.predict(X_test), zero_division=0)

strat_total_return = (test_df['Cum_Strategy_Return'].iloc[-1] / test_df['Cum_Strategy_Return'].iloc[0] - 1) * 100
market_total_return = (test_df['Cum_Market_Return'].iloc[-1] / test_df['Cum_Market_Return'].iloc[0] - 1) * 100

strat_daily_std = test_df['Strategy_Return'].std()
sharpe_ratio = (test_df['Strategy_Return'].mean() / strat_daily_std * np.sqrt(252)) if strat_daily_std > 0 else 0

cum_max = test_df['Portfolio_Strategy'].cummax()
drawdown = (test_df['Portfolio_Strategy'] - cum_max) / cum_max
max_drawdown = drawdown.min() * 100


# --- Tabs Navigation ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price Overview", 
    "📈 Technical Indicators", 
    "🤖 ML Model & Signals", 
    "💰 Backtest Performance", 
    "📋 Data & Export"
])

# --- TAB 1: Price Overview ---
with tab1:
    st.subheader(f"{ticker} Price History & Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    latest_close = df_raw['Close'].iloc[-1]
    prev_close = df_raw['Close'].iloc[-2]
    day_change = ((latest_close - prev_close) / prev_close) * 100
    
    col1.metric("Latest Close Price", f"${latest_close:.2f}", f"{day_change:+.2f}%")
    col2.metric("52-Week High", f"${df_raw['High'].tail(252).max():.2f}")
    col3.metric("52-Week Low", f"${df_raw['Low'].tail(252).min():.2f}")
    col4.metric("Avg Daily Volatility", f"{df_feat['Volatility'].iloc[-1]*100:.2f}%")
    
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df_raw.index, open=df_raw['Open'], high=df_raw['High'],
        low=df_raw['Low'], close=df_raw['Close'], name="Candlestick"
    ))
    fig_price.update_layout(
        title=f"{ticker} Candlestick Price Chart",
        xaxis_title="Date", yaxis_title="Price ($)",
        template="plotly_dark", height=500
    )
    st.plotly_chart(fig_price, use_container_width=True)

# --- TAB 2: Technical Indicators ---
with tab2:
    st.subheader("Technical Analysis Overlay")
    fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig_tech.add_trace(go.Scatter(x=df_feat.index, y=df_feat['Close'], name='Close Price', line=dict(color='white')), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_Short'], name=f'SMA {short_window}', line=dict(color='orange')), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_Long'], name=f'SMA {long_window}', line=dict(color='cyan')), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_feat.index, y=df_feat['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_feat.index, y=df_feat['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash'), fill='tonexty'), row=1, col=1)
    
    fig_tech.add_trace(go.Scatter(x=df_feat.index, y=df_feat['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig_tech.update_layout(height=600, template="plotly_dark", title_text=f"{ticker} Technical Indicators")
    st.plotly_chart(fig_tech, use_container_width=True)

# --- TAB 3: ML Model & Signals ---
with tab3:
    st.subheader("Random Forest Classifier Diagnostics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Out-of-Sample Accuracy", f"{accuracy*100:.2f}%")
    m_col2.metric("Precision", f"{precision:.2f}")
    m_col3.metric("Recall", f"{recall:.2f}")
    m_col4.metric("Train/Test Boundary", test_df.index[0].strftime('%Y-%m-%d'))
    
    col_feat, col_sig = st.columns(2)
    with col_feat:
        st.markdown("##### Feature Importance")
        importances = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
        fig_imp = go.Figure(go.Bar(x=importances['Importance'], y=importances['Feature'], orientation='h', marker_color='teal'))
        fig_imp.update_layout(height=400, template="plotly_dark", title="Random Forest Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_sig:
        st.markdown("##### Recent Predicted Signals")
        recent_signals = df_feat[['Close', 'Predicted_Signal', 'Position']].tail(10)
        recent_signals['Signal'] = recent_signals['Predicted_Signal'].map({1: "🟢 BUY", 0: "🔴 HOLD/CASH"})
        st.dataframe(recent_signals[['Close', 'Signal']], use_container_width=True)

# --- TAB 4: Backtest Performance ---
# Test period predictions & notebook alias mappings
y_test_pred = model.predict(X_test)
test_sig = y_test_pred                 # Notebook predictions alias
test_bt = test_df                      # Notebook backtest DataFrame alias
INITIAL_CAPITAL = initial_capital      # Sidebar capital alias
TRANSACTION_FEE = transaction_cost     # Sidebar fee alias
test_row = {                           # Financial metrics dictionary alias
    "Trades": test_df["Position_Filled"].diff().abs().fillna(test_df["Position_Filled"].abs()).gt(0).sum(),
    "% Days in Market": (test_df["Position_Filled"] != 0).mean()
}
with tab4:
    st.subheader(f"Out-of-Sample Evaluation & Backtest Analysis ({future_period}-Day Horizon Strategy)")
    
    # --- Confusion Matrix & Equity Plot Section ---
    st.markdown("### 📊 Test Confusion Matrix & Equity Curve")
    
    CLASS_NAMES = {-1: "Sell", 0: "Hold", 1: "Buy"}
    COLORS = {"strategy": "#16a34a", "buyhold": "#6b7280"}
    
    # 1x2 Matplotlib/Seaborn Figure
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    
    # Subplot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=[-1, 0, 1])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
        xticklabels=[CLASS_NAMES[c] for c in (-1, 0, 1)], 
        yticklabels=[CLASS_NAMES[c] for c in (-1, 0, 1)]
    )
    axes[0].set_xlabel("Predicted (after threshold)")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion matrix: test")
    
    # Subplot 2: Equity Curve
    axes[1].plot(
        test_df.index, test_df["Portfolio_Strategy"], color=COLORS["strategy"], lw=2,
        label=f"ML strategy (Sharpe {strat_fm['Sharpe']:.2f})"
    )
    axes[1].plot(
        test_df.index, test_df["Portfolio_Market"], color=COLORS["buyhold"], ls="--",
        label=f"Buy & Hold (Sharpe {bh_fm['Sharpe']:.2f})"
    )
    axes[1].set_title(f"Test-period equity, ${initial_capital:,.0f} start, {transaction_cost:.1%} fees")
    axes[1].set_ylabel("Portfolio value ($)")
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # --- Financial Metrics Summary Table Section ---
    st.markdown("### 📑 Test Financial Metrics")
    
    fin_table = pd.DataFrame({"ML Strategy": strat_fm, "Buy & Hold": bh_fm})
    
    styler = fin_table.style
    for r in fin_table.index:
        styler = styler.format(
            {"Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Trades": "{:.0f}"}.get(r, "{:.1%}"),
            subset=pd.IndexSlice[[r], :]
        )
    
    st.dataframe(styler, use_container_width=True)

# --- TAB 5: Data & Export ---
with tab5:
    st.subheader("Engineered Dataset & Model Signals")
    st.dataframe(df_feat.tail(100), use_container_width=True)
    csv_data = df_feat.to_csv().encode('utf-8')
    st.download_button(label="📥 Download Full Backtest Dataset (CSV)", data=csv_data, file_name=f"{ticker}_ml_trading_backtest.csv", mime="text/csv")

st.markdown("---")
st.caption("Disclaimer: This application is for educational and analytical purposes only. Past performance is not indicative of future market results.")