import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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
with tab4:
    st.subheader("Out-of-Sample Backtest Results")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Strategy Total Return", f"{strat_total_return:+.2f}%")
    p_col2.metric("Buy & Hold Return", f"{market_total_return:+.2f}%")
    p_col3.metric("Annualized Sharpe Ratio", f"{sharpe_ratio:.2f}")
    p_col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=test_df.index, y=test_df['Portfolio_Strategy'], name='ML Strategy Portfolio', line=dict(color='springgreen', width=2)))
    fig_equity.add_trace(go.Scatter(x=test_df.index, y=test_df['Portfolio_Market'], name='Buy & Hold TSLA Portfolio', line=dict(color='deepskyblue', width=2, dash='dash')))
    fig_equity.update_layout(
        title=f"Portfolio Value Growth starting from ${initial_capital:,.2f} (Test Period)",
        xaxis_title="Date", yaxis_title="Portfolio Value ($)", template="plotly_dark", height=500
    )
    st.plotly_chart(fig_equity, use_container_width=True)

# --- TAB 5: Data & Export ---
with tab5:
    st.subheader("Engineered Dataset & Model Signals")
    st.dataframe(df_feat.tail(100), use_container_width=True)
    csv_data = df_feat.to_csv().encode('utf-8')
    st.download_button(label="📥 Download Full Backtest Dataset (CSV)", data=csv_data, file_name=f"{ticker}_ml_trading_backtest.csv", mime="text/csv")

st.markdown("---")
st.caption("Disclaimer: This application is for educational and analytical purposes only. Past performance is not indicative of future market results.")