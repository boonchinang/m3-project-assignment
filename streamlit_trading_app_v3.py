"""ML Trading Model Dashboard — built from tsla_trading_model_v3.ipynb.

The ticker is NOT hardcoded anywhere in the display: every title, label,
chart and message uses the ticker typed in the sidebar. The only place the
letters of the default ticker appear is the sidebar input's initial value,
which you can change to any ticker.

What it does (same method as the v3 notebook):
  * Downloads daily OHLCV from Yahoo Finance (split/dividend-adjusted).
    If the download fails it falls back to synthetic data — and says so
    loudly, instead of silently showing fake prices.
  * Builds only the 10 most important of the notebook's 18 scale-free
    technical features (see TOP_10_FEATURES below), labels each day
    Buy (+1) / Hold (0) / Sell (-1) from its future 30-day return,
    and splits chronologically with purging: train / validation / test.
  * Trains the notebook's candidates (Dummy, Logistic Regression,
    Random Forest, Histogram Gradient Boosting, XGBoost) on train,
    scores them on validation, picks the best by Sharpe (or Macro F1),
    refits on train + validation, and scores once on the sealed test set.
  * Backtests with the notebook's rules: signal after the close on day t,
    filled at the close of day t+1 (execution lag 2), 0.1% per unit of
    position change.

Run it with:  streamlit run streamlit_trading_app_v3_top10.py
Install first: pip install -r requirements-v3-top10.txt
"""

import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
CLASS_NAMES = {-1: "Sell (-1)", 0: "Hold (0)", 1: "Buy (1)"}

# ---------------------------------------------------------------------------
# XGBoost wrapper (from the v3 notebook, section 6.1a)
# XGBoost only accepts labels 0..n-1, but this project labels -1/0/1, so a
# plain XGBClassifier fails at .fit(). This wrapper shifts labels +1 for
# training and -1 back on predict, so .classes_ stays [-1, 0, 1].
# Note: XGBClassifier has no class_weight="balanced" option.
# ---------------------------------------------------------------------------
class XGBoostShifted(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, max_depth=3, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, **kw):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
            eval_metric="mlogloss", random_state=self.random_state, n_jobs=self.n_jobs)
        self.model_.fit(X, y + 1, **kw)
        return self

    def predict(self, X, **kw):
        return self.model_.predict(X, **kw) - 1

    def predict_proba(self, X, **kw):
        return self.model_.predict_proba(X, **kw)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


# ---------------------------------------------------------------------------
# Page + sidebar — the ticker comes from here and nowhere else
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ML Trading Model Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.sidebar.header("🛠️ Model & Strategy Controls")
ticker = st.sidebar.text_input("Stock Ticker", value="TSLA").strip().upper()

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Start Date", value=datetime.date(2017, 1, 1))
with col_d2:
    end_date = st.date_input("End Date", value=datetime.date.today())
col_d3, col_d4 = st.sidebar.columns(2)
with col_d3:
    val_start = st.date_input("Validation Starts", value=datetime.date(2021, 1, 1))
with col_d4:
    test_start = st.date_input("Test Starts", value=datetime.date(2023, 1, 1))

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Target Definition")
future_period = st.sidebar.slider("Look-Ahead Horizon (Trading Days)", 1, 60, 30)
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    buy_threshold_pct = st.number_input("Buy Thresh (%)", 0.0, 50.0, 6.0, 0.5)
with col_t2:
    sell_threshold_pct = st.number_input("Sell Thresh (%)", -50.0, 0.0, -6.0, 0.5)
buy_threshold = buy_threshold_pct / 100.0
sell_threshold = sell_threshold_pct / 100.0

allow_shorting = st.sidebar.checkbox("Allow Short Selling (-1 Position)", value=True)
long_only = not allow_shorting

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Machine Learning Model")
model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ["Auto-select best on validation (Sharpe)", "Logistic Regression",
     "Random Forest", "Gradient Boosting", "XGBoost"])
confidence = st.sidebar.slider("Confidence Threshold", 0.34, 0.70, 0.45, 0.01,
    help="A Buy/Sell signal is issued only if its class probability is at least this.")
auto_threshold = st.sidebar.checkbox(
    "Auto-tune confidence threshold on validation", value=False,
    help="Sweeps 0.34–0.70 on validation and picks the threshold with the best Sharpe (min. 6 trades).")
run_tuning = st.sidebar.checkbox(
    "Run hyperparameter tuning (GridSearchCV, slow)", value=False,
    help="Tunes each model with TimeSeriesSplit on the training set only, as in the notebook §7. Takes several minutes.")

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Backtest Settings")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=1000)
commission = st.sidebar.number_input("Transaction Fee (%)", value=0.1, step=0.05) / 100.0
execution_lag = st.sidebar.number_input("Execution Lag (days)", value=2, min_value=1, max_value=5, step=1,
    help="Signal after the close on day t is filled at the close of day t+1 (lag 2 in the notebook).")

if not ticker:
    st.warning("Enter a stock ticker in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Header — dynamic ticker
# ---------------------------------------------------------------------------
st.header(f"⚡ {ticker} ML Trading Model Dashboard")
st.markdown(
    "<p style='font-size:1.1rem;color:#6c757d'>Algorithmic Trading, "
    "Feature Engineering, Custom Target Horizon &amp; Thresholds, "
    "Machine Learning Signals &amp; Backtesting</p>",
    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data: Yahoo Finance, with a labelled synthetic fallback
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start, end):
    """Returns (df, source_label, is_synthetic)."""
    try:
        import yfinance as yf
        df = yf.download(symbol, start=str(start), end=str(end),
                         auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            if not df.empty:
                return df, "Yahoo Finance", False
    except Exception:
        pass
    # Synthetic fallback — only for testing the code, clearly labelled.
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.bdate_range(start, end, inclusive="left")
    rets = rng.normal(0.0008, 0.018, len(dates))
    close = 15.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.008, len(dates))))
    low = close * (1 - np.abs(rng.normal(0, 0.008, len(dates))))
    open_ = low + (high - low) * rng.uniform(0.1, 0.9, len(dates))
    vol = rng.integers(40_000_000, 120_000_000, len(dates))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                       "Close": close, "Volume": vol},
                      index=pd.DatetimeIndex(dates, name="Date"))
    return df, "SYNTHETIC (offline fallback)", True


raw_df, data_source, is_synthetic = load_stock_data(ticker, start_date, end_date)
if raw_df.empty:
    st.error(f"No data found for symbol '{ticker}' and the selected date range.")
    st.stop()
if is_synthetic:
    st.warning(
        f"⚠️ Could not download real data for **{ticker}** — showing **SYNTHETIC** "
        "data instead. Results below are for testing the code only. "
        "Check the ticker and your internet connection, then re-run.")
else:
    st.caption(f"Data source: {data_source} · {len(raw_df):,} trading days · "
               f"{raw_df.index.min():%Y-%m-%d} to {raw_df.index.max():%Y-%m-%d}")

# ---------------------------------------------------------------------------
# Features — the 10 most important of the v3 notebook's 18 features
# ---------------------------------------------------------------------------
# Ranking method (reproduced from the v3 notebook, §8.1): permutation
# importance on the 2021-2022 validation set for the notebook's best model
# (Logistic Regression, tuned C=1.0), scoring macro F1, n_repeats=10,
# on real Yahoo Finance data for the default ticker, 2012-2026.
# The top 5 reproduced the notebook's own Top 5 exactly. Full ranking:
#   1 MACD_Pct            +0.033   6 Volume_Ratio       +0.001
#   2 MACD_Signal_Pct     +0.028   7 Return_1D          +0.001
#   3 Dist_52W_High       +0.023   8 HL_Range_14        -0.003
#   4 Vol_Ratio_20_60     +0.020   9 Return_5D          -0.004
#   5 Ratio_SMA50         +0.007  10 Return_63D         -0.004
# Honest caveat: only the top 5 are clearly positive — ranks 6-10 are
# essentially zero (within noise). Importance is also model-dependent:
# Random Forest's own importances rank Ratio_SMA200 and Vol_20D highly,
# but those two were dropped here because they scored lowest for the
# notebook's best (Logistic Regression) model. Dropping the other 8
# features therefore loses little for this model, but re-run the
# permutation-importance check if you change ticker, dates or best model.
TOP_10_FEATURES = ["MACD_Pct", "MACD_Signal_Pct", "Dist_52W_High",
                   "Vol_Ratio_20_60", "Ratio_SMA50", "Volume_Ratio",
                   "Return_1D", "HL_Range_14", "Return_5D", "Return_63D"]
FEATURE_COLS = TOP_10_FEATURES


def build_features(df):
    """Only the 10 features in FEATURE_COLS are built (same formulas as v3)."""
    d = df.copy()
    c = d["Close"]
    for n in (1, 5, 63):
        d[f"Return_{n}D"] = c.pct_change(n)
    d["Ratio_SMA50"] = c / c.rolling(50).mean()
    d["Dist_52W_High"] = c / c.rolling(252).max() - 1
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    d["MACD_Pct"], d["MACD_Signal_Pct"] = macd / c, signal / c
    r = c.pct_change()
    d["Vol_20D"] = r.rolling(20).std()
    d["Vol_Ratio_20_60"] = d["Vol_20D"] / (r.rolling(60).std() + 1e-8)
    d["HL_Range_14"] = ((d["High"] - d["Low"]) / c).rolling(14).mean()
    d["Volume_Ratio"] = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-8)
    return d


def create_target(df, horizon, buy_thresh, sell_thresh):
    data = df.copy()
    data["Future_Return"] = data["Close"].shift(-horizon) / data["Close"] - 1.0
    data = data.dropna(subset=["Future_Return"])
    data["Target"] = np.select([data["Future_Return"] > buy_thresh,
                                data["Future_Return"] < sell_thresh], [1, -1], default=0)
    return data


df_model = create_target(build_features(raw_df), future_period, buy_threshold, sell_threshold)
df_model = df_model.dropna(subset=FEATURE_COLS)


def purged_index(df, end, start=None, purge=30):
    idx = df.index[df.index < pd.Timestamp(end)]
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    return idx[:-purge] if purge > 0 else idx


train_idx = purged_index(df_model, val_start, purge=future_period)
val_idx = purged_index(df_model, test_start, start=val_start, purge=future_period)
test_idx = df_model.index[df_model.index >= pd.Timestamp(test_start)]
trval_idx = purged_index(df_model, test_start, purge=future_period)

if min(len(train_idx), len(val_idx), len(test_idx)) < 50:
    st.error(
        f"Not enough data for **{ticker}** with these split dates — "
        f"train {len(train_idx)}, validation {len(val_idx)}, test {len(test_idx)} rows. "
        "The ticker may have a shorter history than the split dates assume: "
        "move Validation/Test Starts later, or Start Date earlier.")
    st.stop()
if set(df_model.loc[train_idx, "Target"].unique()) != {-1, 0, 1}:
    st.error(f"Training data for **{ticker}** does not contain all three classes "
             "(Sell/Hold/Buy). Try a longer history or different thresholds.")
    st.stop()

X_train = df_model.loc[train_idx, FEATURE_COLS]
y_train = df_model.loc[train_idx, "Target"]
X_val = df_model.loc[val_idx, FEATURE_COLS]
y_val = df_model.loc[val_idx, "Target"]
X_test = df_model.loc[test_idx, FEATURE_COLS]
y_test = df_model.loc[test_idx, "Target"]


# ---------------------------------------------------------------------------
# Backtest helpers — from the v3 notebook (§6)
# ---------------------------------------------------------------------------
def signals_from_proba(proba, classes, threshold):
    classes = list(classes)
    p_buy, p_sell = proba[:, classes.index(1)], proba[:, classes.index(-1)]
    sig = np.zeros(len(proba), dtype=int)
    sig[(p_buy >= threshold) & (p_buy > p_sell)] = 1
    sig[(p_sell >= threshold) & (p_sell > p_buy)] = -1
    return sig


def run_backtest(close, signals, lag, fee, long_only_flag, capital):
    bt = pd.DataFrame({"Close": close.values, "Signal": np.asarray(signals)}, index=close.index)
    if long_only_flag:
        bt["Signal"] = bt["Signal"].clip(lower=0)
    bt["Position"] = bt["Signal"].shift(lag).fillna(0)
    bt["Stock_Return"] = bt["Close"].pct_change().fillna(0)
    bt["Turnover"] = bt["Position"].diff().abs().fillna(bt["Position"].abs())
    bt["Strategy_Return"] = bt["Position"] * bt["Stock_Return"] - bt["Turnover"] * fee
    bt["BuyHold_Return"] = bt["Stock_Return"]
    bt.iloc[0, bt.columns.get_loc("BuyHold_Return")] -= fee
    bt["Strategy_Equity"] = capital * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = capital * (1 + bt["BuyHold_Return"]).cumprod()
    return bt


def financial_metrics(r, capital, rf=0.02):
    r = pd.Series(r).fillna(0)
    equity = capital * (1 + r).cumprod()
    years = len(r) / 252
    sd = r.std()
    excess = r - rf / 252
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return {
        "Total Return": equity.iloc[-1] / capital - 1,
        "CAGR": (equity.iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else np.nan,
        "Volatility": sd * np.sqrt(252),
        "Sharpe": excess.mean() / sd * np.sqrt(252) if sd > 1e-12 else np.nan,
        "Sortino": excess.mean() / downside * np.sqrt(252) if downside > 1e-12 else np.nan,
        "Max Drawdown": (equity / equity.cummax() - 1).min(),
    }


def evaluate(model, X, y, idx, threshold, name=""):
    proba = model.predict_proba(X)
    sig = signals_from_proba(proba, model.classes_, threshold)
    bt = run_backtest(df_model.loc[idx, "Close"], sig, execution_lag,
                      commission, long_only, initial_capital)
    fm = financial_metrics(bt["Strategy_Return"], initial_capital)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro",
                            labels=list(model.classes_))
    except ValueError:
        auc = np.nan
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y, sig),
        "Balanced Acc": balanced_accuracy_score(y, sig),
        "Macro F1": f1_score(y, sig, average="macro", labels=[-1, 0, 1], zero_division=0),
        "Buy Precision": precision_score(y, sig, labels=[1], average="macro", zero_division=0),
        "Sell Precision": precision_score(y, sig, labels=[-1], average="macro", zero_division=0),
        "ROC AUC (OvR)": auc,
        "Sharpe": fm["Sharpe"], "CAGR": fm["CAGR"], "Max Drawdown": fm["Max Drawdown"],
        "Trades": int((bt["Turnover"] > 0).sum()),
        "% Days in Market": (bt["Position"] != 0).mean(),
    }
    return row, sig, proba, bt


def buyhold_row(idx, name="Buy & Hold (current process)"):
    bt = run_backtest(df_model.loc[idx, "Close"], np.ones(len(idx), dtype=int),
                      execution_lag, commission, long_only, initial_capital)
    fm = financial_metrics(bt["BuyHold_Return"], initial_capital)
    return {"Model": name, "Sharpe": fm["Sharpe"], "CAGR": fm["CAGR"],
            "Max Drawdown": fm["Max Drawdown"], "Trades": 1, "% Days in Market": 1.0}


# ---------------------------------------------------------------------------
# Train candidates, score on validation, pick the best
# ---------------------------------------------------------------------------
candidate_models = {
    "Majority class (Dummy)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=25,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200,
        class_weight="balanced", random_state=RANDOM_STATE),
    "XGBoost": XGBoostShifted(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1),
}

search_spaces = {
    "Logistic Regression": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
        {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                               random_state=RANDOM_STATE, n_jobs=1),
        {"max_depth": [3, 4, 6, 8], "min_samples_leaf": [10, 25, 50],
         "max_features": ["sqrt", 0.5]}),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 5],
         "max_iter": [100, 300], "l2_regularization": [0.0, 1.0]}),
    "XGBoost": (
        XGBoostShifted(random_state=RANDOM_STATE, n_jobs=-1),
        {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 5],
         "n_estimators": [100, 300], "subsample": [0.8, 1.0]}),
}

with st.spinner(f"Training models for {ticker}…"):
    if run_tuning:
        tscv = TimeSeriesSplit(n_splits=5, gap=future_period)
        fitted = {}
        val_rows = [buyhold_row(val_idx)]
        for name, (est, grid) in search_spaces.items():
            gs = GridSearchCV(est, grid, cv=tscv, scoring="f1_macro",
                              n_jobs=-1, return_train_score=True)
            gs.fit(X_train, y_train)
            fitted[name] = gs.best_estimator_
            val_rows.append(evaluate(gs.best_estimator_, X_val, y_val, val_idx,
                                     confidence, f"{name} (tuned)")[0])
        fitted["Majority class (Dummy)"] = clone(
            candidate_models["Majority class (Dummy)"]).fit(X_train, y_train)
        val_rows.insert(1, evaluate(fitted["Majority class (Dummy)"], X_val, y_val,
                                    val_idx, confidence, "Majority class (Dummy)")[0])
        val_table = pd.DataFrame(val_rows).set_index("Model")
        ml_rows = val_table[val_table.index.str.contains("tuned")]
        key = "Sharpe" if ml_rows["Sharpe"].notna().any() else "Macro F1"
        best_label = ml_rows[key].idxmax()
        best_name = best_label.replace(" (tuned)", "")
        best_model_trainfit = fitted[best_name]
    else:
        fitted = {}
        val_rows = [buyhold_row(val_idx)]
        for name, model in candidate_models.items():
            m = clone(model).fit(X_train, y_train)
            fitted[name] = m
            val_rows.append(evaluate(m, X_val, y_val, val_idx, confidence, name)[0])
        val_table = pd.DataFrame(val_rows).set_index("Model")
        ml_rows = val_table.drop(index="Buy & Hold (current process)", errors="ignore")
        ml_rows = ml_rows.drop(index="Majority class (Dummy)", errors="ignore")
        key = "Sharpe" if ml_rows["Sharpe"].notna().any() else "Macro F1"
        best_label = ml_rows[key].idxmax()
        best_name = best_label
        best_model_trainfit = fitted[best_name]

    if model_choice != "Auto-select best on validation (Sharpe)":
        best_name = model_choice
        best_label = f"{model_choice} (chosen in sidebar)"
        best_model_trainfit = fitted[best_name]

    # Auto-tune the confidence threshold on validation, for the chosen model
    best_threshold = confidence
    if auto_threshold:
        val_proba = best_model_trainfit.predict_proba(X_val)
        sweep = []
        for th in np.round(np.arange(0.34, 0.70, 0.02), 2):
            sig = signals_from_proba(val_proba, best_model_trainfit.classes_, th)
            bt = run_backtest(df_model.loc[val_idx, "Close"], sig, execution_lag,
                              commission, long_only, initial_capital)
            fm = financial_metrics(bt["Strategy_Return"], initial_capital)
            sweep.append({"Threshold": th, "Sharpe": fm["Sharpe"], "CAGR": fm["CAGR"],
                          "Macro F1": f1_score(y_val, sig, average="macro",
                                               labels=[-1, 0, 1], zero_division=0),
                          "Trades": int((bt["Turnover"] > 0).sum())})
        sweep_df = pd.DataFrame(sweep)
        eligible = sweep_df[(sweep_df["Trades"] >= 6) & sweep_df["Sharpe"].notna()]
        if len(eligible):
            best_threshold = float(eligible.loc[eligible["Sharpe"].idxmax(), "Threshold"])

    # Refit the chosen model once on train + validation, score on sealed test
    final_model = clone(best_model_trainfit).fit(
        df_model.loc[trval_idx, FEATURE_COLS], df_model.loc[trval_idx, "Target"])
    test_row, test_sig, test_proba, test_bt = evaluate(
        final_model, X_test, y_test, test_idx, best_threshold, best_label)

# ---------------------------------------------------------------------------
# Top summary metrics — dynamic ticker
# ---------------------------------------------------------------------------
latest_close = float(raw_df["Close"].iloc[-1])
prev_close = float(raw_df["Close"].iloc[-2])
pct_change = (latest_close - prev_close) / prev_close * 100
acc = accuracy_score(y_test, test_sig)
strat_total_return = ((test_bt["Strategy_Equity"].iloc[-1] - initial_capital) / initial_capital) * 100
bench_total_return = ((test_bt["BuyHold_Equity"].iloc[-1] - initial_capital) / initial_capital) * 100

st.info(
    f"🎯 **Target Configuration Active for {ticker}**: Look-Ahead Horizon = "
    f"**{future_period} Trading Days** | Buy Threshold (1) = **> +{buy_threshold_pct:.1f}%** | "
    f"Sell Threshold (-1) = **< {sell_threshold_pct:.1f}%** | Neutral (0) = **Between thresholds** | "
    f"Selected model: **{best_name}** | Confidence threshold: **{best_threshold:.2f}**"
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"Latest {ticker} Close", f"${latest_close:.2f}", f"{pct_change:+.2f}%")
m2.metric("ML Model Accuracy (test)", f"{acc * 100:.1f}%")
m3.metric("Strategy Return (test)", f"{strat_total_return:+.1f}%")
m4.metric("Buy & Hold Return (test)", f"{bench_total_return:+.1f}%")
m5.metric("Final Portfolio (test)", f"${test_bt['Strategy_Equity'].iloc[-1]:,.2f}")

st.markdown("---")
st.caption(f"Splits for {ticker}: train {train_idx.min():%Y-%m-%d}–{train_idx.max():%Y-%m-%d} "
           f"({len(train_idx):,} rows) · validation {val_idx.min():%Y-%m-%d}–{val_idx.max():%Y-%m-%d} "
           f"({len(val_idx):,} rows) · test {test_idx.min():%Y-%m-%d}–{test_idx.max():%Y-%m-%d} "
           f"({len(test_idx):,} rows) · 10 features (top 10 by permutation importance) · data: {data_source}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price & Indicators",
    "🤖 ML Model Analysis",
    "📈 Strategy Backtest",
    "🔮 Live Signal Sandbox",
])

# --- TAB 1: price & indicators ---
with tab1:
    st.subheader(f"Price History & Technical Overlays — {ticker}")
    feat_df = df_model[FEATURE_COLS].copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Price & Moving Averages",
                                        "Relative Strength Index (RSI)", "MACD"),
                        row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["Close"], name="Close Price",
                             line=dict(color="#E82127", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["Close"].rolling(10).mean(),
                             name="SMA 10", line=dict(color="#0066CC", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["Close"].rolling(50).mean(),
                             name="SMA 50", line=dict(color="#FF9900", width=1)), row=1, col=1)
    # RSI is chart-only here (it is NOT one of the 10 model features).
    _delta = raw_df["Close"].diff()
    _gain = _delta.clip(lower=0).rolling(14).mean()
    _loss = (-_delta.clip(upper=0)).rolling(14).mean()
    _rsi = 100 - 100 / (1 + _gain / (_loss + 1e-8))
    fig.add_trace(go.Scatter(x=raw_df.index, y=_rsi, name="RSI (14)",
                             line=dict(color="#8884d8")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    macd = raw_df["Close"].ewm(span=12, adjust=False).mean() - raw_df["Close"].ewm(span=26, adjust=False).mean()
    fig.add_trace(go.Scatter(x=raw_df.index, y=macd, name="MACD", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=raw_df.index, y=macd.ewm(span=9, adjust=False).mean(),
                             name="Signal", line=dict(color="orange")), row=3, col=1)
    fig.update_layout(height=700, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Latest {ticker} features (last labelled day {df_model.index.max():%Y-%m-%d}):")
    st.dataframe(df_model.loc[[df_model.index.max()], FEATURE_COLS].T.rename(
        columns={df_model.index.max(): "Value"}), use_container_width=True)

# --- TAB 2: model analysis ---
with tab2:
    st.subheader(f"Model Evaluation: {best_name} — {ticker}")
    st.write("#### Validation comparison (all candidates, same threshold)")
    st.dataframe(val_table.style.format(
        {c: "{:.1%}" for c in ["Accuracy", "Balanced Acc", "Macro F1",
                               "Buy Precision", "Sell Precision", "CAGR",
                               "Max Drawdown", "% Days in Market"] if c in val_table.columns},
        na_rep="–"), use_container_width=True)
    st.caption(f"Selected on validation by {key}: **{best_label}**. "
               "Test results below are the single sealed-test score of the refitted model.")

    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Confusion Matrix (test)")
        cm = confusion_matrix(y_test, test_sig, labels=[-1, 0, 1])
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                           x=["Sell (-1)", "Neutral (0)", "Buy (1)"],
                           y=["Sell (-1)", "Neutral (0)", "Buy (1)"],
                           color_continuous_scale="Reds")
        fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
    with c2:
        st.write("#### Feature Importance (test model)")
        importances = None
        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_
        elif hasattr(final_model, "coef_"):
            importances = np.abs(final_model.coef_).mean(axis=0)
        elif hasattr(final_model, "named_steps"):
            clf = final_model.named_steps.get("clf")
            if clf is not None and hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_).mean(axis=0)
        if importances is not None:
            feat_imp = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances}
                                    ).sort_values("Importance", ascending=True)
            fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Viridis")
            fig_imp.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model type.")

    st.write("#### Detailed Classification Metrics — test (weighted)")
    prec = precision_score(y_test, test_sig, average="weighted", zero_division=0)
    rec = recall_score(y_test, test_sig, average="weighted", zero_division=0)
    f1 = f1_score(y_test, test_sig, average="weighted", zero_division=0)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Precision (Weighted)", f"{prec:.3f}")
    mc2.metric("Recall (Weighted)", f"{rec:.3f}")
    mc3.metric("F1 Score (Weighted)", f"{f1:.3f}")
    mc4.metric("Test Samples", f"{len(y_test)}")
    st.text(classification_report(y_test, test_sig, labels=[-1, 0, 1],
                                  target_names=["Sell (-1)", "Neutral (0)", "Buy (1)"],
                                  zero_division=0))

# --- TAB 3: backtest ---
with tab3:
    st.subheader(f"Algorithmic Trading Backtest Performance — {ticker}")
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt["Strategy_Equity"],
                                name="ML Strategy", line=dict(color="#00CC66", width=2.5)))
    fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt["BuyHold_Equity"],
                                name=f"Buy & Hold {ticker}",
                                line=dict(color="#E82127", width=1.5, dash="dash")))
    fig_bt.update_layout(title="Portfolio Growth ($)", yaxis_title="Portfolio Value ($)",
                         height=450, template="plotly_white", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig_bt, use_container_width=True)

    strat_fm = financial_metrics(test_bt["Strategy_Return"], initial_capital)
    bh_fm = financial_metrics(test_bt["BuyHold_Return"], initial_capital)
    win_rate = (test_bt["Strategy_Return"] > 0).mean() * 100
    st.write("#### Performance Summary (test)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Sharpe Ratio (Annualized)", f"{strat_fm['Sharpe']:.2f}")
    b2.metric("Max Drawdown", f"{strat_fm['Max Drawdown'] * 100:.2f}%")
    b3.metric("Daily Win Rate", f"{win_rate:.1f}%")
    b4.metric("Total Trades Executed", f"{int(np.abs(test_bt['Position'].diff()).sum())}")
    st.caption(f"Strategy Sharpe {strat_fm['Sharpe']:.2f} vs Buy & Hold Sharpe "
               f"{bh_fm['Sharpe']:.2f} · Strategy CAGR {strat_fm['CAGR']:.1%} vs "
               f"Buy & Hold CAGR {bh_fm['CAGR']:.1%} · Sortino {strat_fm['Sortino']:.2f}")

# --- TAB 4: live signal sandbox ---
with tab4:
    st.subheader(f"🔮 Real-Time Trading Signal Predictor — {ticker}")
    st.write("The sliders start at the latest day's actual feature values for "
             f"{ticker}. Adjust them to test what signal the trained model would "
             "give for that hypothetical combination:")
    latest_feat = build_features(raw_df).dropna(subset=FEATURE_COLS).iloc[-1]
    latest_date = build_features(raw_df).dropna(subset=FEATURE_COLS).index[-1]
    st.caption(f"Latest features as of {latest_date:%Y-%m-%d} (close "
               f"${float(raw_df['Close'].iloc[-1]):.2f}). These are inputs only — "
               "the sandbox does not pull live indicators by itself.")

    # Slider ranges from the training data (1st–99th percentile, widened a little)
    train_feat = df_model.loc[train_idx, FEATURE_COLS]
    inputs = {}
    groups = [("Momentum", ["Return_1D", "Return_5D", "Return_63D"]),
              ("Trend", ["Ratio_SMA50", "Dist_52W_High"]),
              ("Oscillators", ["MACD_Pct", "MACD_Signal_Pct"]),
              ("Volatility & Volume", ["Vol_Ratio_20_60", "HL_Range_14", "Volume_Ratio"])]
    for group_name, cols in groups:
        st.write(f"**{group_name}**")
        gcols = st.columns(3)
        for j, feat in enumerate(cols):
            lo = float(train_feat[feat].quantile(0.01))
            hi = float(train_feat[feat].quantile(0.99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(latest_feat[feat]) - 1.0, float(latest_feat[feat]) + 1.0
            pad = (hi - lo) * 0.1
            lo, hi = lo - pad, hi + pad
            default = float(np.clip(latest_feat[feat], lo, hi))
            step = (hi - lo) / 100
            with gcols[j % 3]:
                inputs[feat] = st.slider(feat, lo, hi, default, step=step,
                                         key=f"sandbox_{feat}")

    input_df = pd.DataFrame([inputs], columns=FEATURE_COLS)
    sandbox_proba = final_model.predict_proba(input_df)[0]
    sandbox_sig = signals_from_proba(
        sandbox_proba.reshape(1, -1), final_model.classes_, best_threshold)[0]
    proba_map = {int(c): float(p) for c, p in zip(final_model.classes_, sandbox_proba)}

    st.markdown("---")
    st.write("### Predicted Target Signal:")
    if sandbox_sig == 1:
        st.success(f"🟢 **BUY SIGNAL (1)** — Expecting future return > +{buy_threshold_pct:.1f}% "
                   f"over the next {future_period} trading days.")
    elif sandbox_sig == -1:
        st.error(f"🔴 **SELL / SHORT SIGNAL (-1)** — Expecting future return < {sell_threshold_pct:.1f}% "
                 f"over the next {future_period} trading days.")
    else:
        st.warning(f"🟡 **NEUTRAL / HOLD SIGNAL (0)** — Expecting price movement between "
                   f"{sell_threshold_pct:.1f}% and +{buy_threshold_pct:.1f}%.")
    st.write(f"Model probabilities — Buy: {proba_map.get(1, 0):.1%} · "
             f"Hold: {proba_map.get(0, 0):.1%} · Sell: {proba_map.get(-1, 0):.1%} "
             f"(confidence threshold {best_threshold:.2f}, model: {best_name})")

st.caption(f"Educational project only — not investment advice. "
           f"Model: {best_name} for {ticker} · Data: {data_source} · "
           f"Features: 10 (top 10 by permutation importance, v3 notebook) · Execution lag: {execution_lag} day(s).")
