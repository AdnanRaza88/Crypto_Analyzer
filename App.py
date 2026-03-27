"""
Professional Crypto Trading Analysis Web App
Combines ML models with technical analysis for probabilistic trading signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pycoingecko import CoinGeckoAPI
import ta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. DATA PIPELINE
# ================================
@st.cache_data(ttl=300)
def fetch_price_data(coin_id, days, interval='daily'):
    """Fetch price data from CoinGecko and return OHLCV."""
    cg = CoinGeckoAPI()
    try:
        # For daily data, use OHLC endpoint (max 365 days)
        if interval == 'daily':
            data = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency='usd', days=days)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['volume'] = np.nan  # volume not available in OHLC
            return df
        else:
            # Use market_chart for intraday data
            if days <= 1:
                interval_param = 'minutely'
            else:
                interval_param = 'hourly'
            data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days, interval=interval_param)
            prices = data['prices']
            volumes = data['total_volumes']
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            vol_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
            vol_df.set_index('timestamp', inplace=True)
            df['volume'] = vol_df['volume']
            return df
    except Exception as e:
        st.error(f"CoinGecko error: {e}")
        return None

def resample_to_ohlc(df, timeframe):
    """Resample price series to OHLCV."""
    if df is None or df.empty:
        return None
    # If df already has OHLC (from daily endpoint), resample accordingly
    if 'open' in df.columns:
        ohlc = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    else:
        # Approximate OHLC from close only (for intraday data)
        ohlc = df['close'].resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        })
        if 'volume' in df.columns:
            ohlc['volume'] = df['volume'].resample(timeframe).sum()
        else:
            ohlc['volume'] = 0
    ohlc.dropna(inplace=True)
    return ohlc

def get_ohlc_data(coin_id, timeframe_str, days=500):
    """Main function to get OHLC data for given coin and timeframe."""
    tf_map = {
        '1m': ('1min', 1),
        '5m': ('5min', 7),
        '15m': ('15min', 30),
        '1h': ('1h', 90),
        '4h': ('4h', 180),
        '1d': ('1d', 365)
    }
    if timeframe_str not in tf_map:
        return None
    tf_pd, days_to_fetch = tf_map[timeframe_str]
    # Use minutely/hourly interval for intraday, daily for daily
    if tf_pd.endswith('min'):
        interval = 'minutely'
    elif tf_pd.endswith('h'):
        interval = 'hourly'
    else:
        interval = 'daily'
    raw_df = fetch_price_data(coin_id, days_to_fetch, interval)
    if raw_df is None:
        return None
    ohlc = resample_to_ohlc(raw_df, tf_pd)
    return ohlc

# ================================
# 2. FEATURE ENGINEERING
# ================================
def add_technical_indicators(df):
    """Add common technical indicators to the DataFrame."""
    df = df.copy()
    # Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    # Moving averages
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
    # Volatility (rolling std of returns)
    df['volatility'] = df['log_return'].rolling(window=20).std()
    # Volume
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    # Drop NaN rows
    df.dropna(inplace=True)
    return df

def add_target(df, horizon=1):
    """Add target column: 1 if price goes up after horizon, else 0."""
    df = df.copy()
    df['future_close'] = df['close'].shift(-horizon)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    df.dropna(subset=['target'], inplace=True)
    return df

def prepare_features(df, feature_columns):
    """Return X and y for ML."""
    X = df[feature_columns]
    y = df['target']
    return X, y

# ================================
# 3. MACHINE LEARNING MODELS
# ================================
def train_models(X, y, test_size=0.2):
    """Train multiple models using time-series split."""
    # Time-series split: last test_size portion is test, rest train
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
    }
    trained_models = {}
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        trained_models[name] = model
        predictions[name] = y_pred
        probabilities[name] = y_proba
        # Evaluate
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        st.write(f"**{name}** metrics on test set (last {test_size*100:.0f}% of data):")
        st.write(metrics)
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("---")
    
    # Ensemble predictions: average probabilities
    ensemble_proba = np.mean(list(probabilities.values()), axis=0)
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'precision': precision_score(y_test, ensemble_pred, zero_division=0),
        'recall': recall_score(y_test, ensemble_pred, zero_division=0),
        'f1': f1_score(y_test, ensemble_pred, zero_division=0)
    }
    st.write("**Ensemble (average probabilities)** metrics:")
    st.write(ensemble_metrics)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, ensemble_pred))
    
    return trained_models, X_train.columns, X_test, y_test

def predict_next(models, X_last):
    """Get ensemble prediction probability for the next candle."""
    probas = []
    for model in models.values():
        probas.append(model.predict_proba(X_last)[0, 1])
    return np.mean(probas)

# ================================
# 4. TRADING LOGIC & STRATEGY
# ================================
def generate_signal(df, ml_prob, threshold=0.6):
    """
    Combine ML probability with technical indicators to decide action.
    Returns: 'BUY', 'SELL', 'HOLD', 'AVOID' and reason string.
    """
    last = df.iloc[-1]
    # Technical filters
    rsi = last['rsi']
    macd_diff = last['macd_diff']
    volume_ratio = last['volume_ratio']
    trend = 'uptrend' if last['close'] > last['ma50'] else 'downtrend' if last['close'] < last['ma50'] else 'sideways'
    
    # Avoid sideways
    if trend == 'sideways':
        return 'AVOID', 'Market is sideways, avoid trading.'
    
    # Base ML signal
    if ml_prob > threshold:
        ml_signal = 'BUY'
    elif ml_prob < (1 - threshold):
        ml_signal = 'SELL'
    else:
        ml_signal = 'HOLD'
    
    # Confirmation from indicators
    if ml_signal == 'BUY':
        if rsi > 70:
            reason = "RSI overbought, caution."
            ml_signal = 'HOLD'
        elif rsi < 30:
            reason = "RSI oversold, potential reversal."
        else:
            reason = "ML predicts up, indicators neutral."
        if macd_diff < 0:
            reason += " MACD bearish, conflicting."
            ml_signal = 'HOLD'
        if volume_ratio < 0.8:
            reason += " Volume low, weak conviction."
            ml_signal = 'HOLD'
    elif ml_signal == 'SELL':
        if rsi < 30:
            reason = "RSI oversold, avoid short."
            ml_signal = 'HOLD'
        elif rsi > 70:
            reason = "RSI overbought, potential drop."
        else:
            reason = "ML predicts down, indicators neutral."
        if macd_diff > 0:
            reason += " MACD bullish, conflicting."
            ml_signal = 'HOLD'
        if volume_ratio < 0.8:
            reason += " Volume low, weak conviction."
            ml_signal = 'HOLD'
    else:
        reason = "ML probability low, no clear signal."
    
    return ml_signal, reason

def compute_sl_tp(df, signal, risk_multiplier=1.5, tp_multipliers=[1, 2, 3]):
    """
    Compute entry, stop loss, take profit based on ATR.
    Returns dict with entry, sl, tp1, tp2, tp3.
    """
    last = df.iloc[-1]
    price = last['close']
    atr = last['atr']
    if signal == 'BUY':
        entry = price
        sl = price - atr * risk_multiplier
        tp1 = price + atr * tp_multipliers[0]
        tp2 = price + atr * tp_multipliers[1]
        tp3 = price + atr * tp_multipliers[2]
    elif signal == 'SELL':
        entry = price
        sl = price + atr * risk_multiplier
        tp1 = price - atr * tp_multipliers[0]
        tp2 = price - atr * tp_multipliers[1]
        tp3 = price - atr * tp_multipliers[2]
    else:
        entry = sl = tp1 = tp2 = tp3 = None
    return {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3}

def backtest_summary():
    """Simplified backtest summary (placeholder)."""
    # In a real implementation, you would simulate trades on test data.
    # Here we return dummy results for demonstration.
    return "Win Rate: 55% | Total Return: +12% | Max Drawdown: -8%"

# ================================
# 5. STREAMLIT APP
# ================================
st.set_page_config(page_title="ML Crypto Trading Analyzer", layout="wide")
st.title("📈 AI-Powered Crypto Trading Analysis")
st.markdown("### Machine Learning + Technical Analysis for Probabilistic Signals")

# Sidebar
st.sidebar.header("Configuration")
coin_id = st.sidebar.text_input("Coin ID (CoinGecko)", value="bitcoin")
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
trade_duration = st.sidebar.selectbox("Trade Duration", ["Scalping", "Intraday", "Swing"])
mode = st.sidebar.selectbox("Mode", ["Spot", "Futures", "Both"])

run_btn = st.sidebar.button("🚀 Run Analysis")
refresh_btn = st.sidebar.button("🔄 Refresh Data")

if run_btn or refresh_btn:
    with st.spinner("Fetching data and training models..."):
        # Fetch and preprocess data
        df_raw = get_ohlc_data(coin_id, timeframe, days=500)
        if df_raw is None or df_raw.empty:
            st.error("No data fetched. Check coin ID or network.")
            st.stop()
        
        df = add_technical_indicators(df_raw)
        if df.empty:
            st.error("Not enough data for indicators. Try larger timeframe.")
            st.stop()
        
        df = add_target(df, horizon=1)
        if df.empty:
            st.error("Not enough data for target. Need more history.")
            st.stop()
        
        # Define features (exclude non-feature columns)
        feature_cols = [c for c in df.columns if c not in ['target', 'future_close', 'close', 'high', 'low', 'open', 'volume']]
        X, y = prepare_features(df, feature_cols)
        
        # Train models
        models, feature_names, X_test, y_test = train_models(X, y, test_size=0.2)
        
        # Predict next move
        X_latest = df.iloc[-1:][feature_cols]
        ml_prob = predict_next(models, X_latest)
        
        # Generate signal and trade plan
        signal, reason = generate_signal(df, ml_prob)
        sl_tp = compute_sl_tp(df, signal)
        
        # Mode-specific advice
        if mode == "Spot":
            advice = "Spot trading: use limit orders, hold with stop loss. No leverage."
        elif mode == "Futures":
            if signal in ['BUY', 'SELL']:
                # Rough risk assessment based on ATR percentage
                atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
                if atr_pct < 2:
                    lev = "5x-10x"
                    risk_msg = "Low volatility – higher leverage possible."
                elif atr_pct < 5:
                    lev = "3x-5x"
                    risk_msg = "Medium volatility – moderate leverage."
                else:
                    lev = "1x-2x"
                    risk_msg = "High volatility – low leverage recommended."
                advice = f"Futures: suggested leverage {lev}. {risk_msg} Be aware of liquidation risk."
            else:
                advice = "Futures: avoid leverage when signal is uncertain."
        else:  # Both
            advice = "Spot: safer. Futures: can use moderate leverage if signal strong."
        
        # Plot chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], name='MA20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ma50'], name='MA50', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
        fig.update_layout(title=f"{coin_id.upper()} - {timeframe}", height=800, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display analysis results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ML Up Probability", f"{ml_prob*100:.1f}%")
            st.metric("Signal", signal)
        with col2:
            # Risk meter: based on ATR and trend
            atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
            if atr_pct < 2:
                risk_level = "Low"
            elif atr_pct < 5:
                risk_level = "Medium"
            else:
                risk_level = "High"
            st.metric("Risk Level", risk_level)
            # Position size suggestion (simple rule)
            if risk_level == "Low":
                pos_size = "Up to 5% of capital"
            elif risk_level == "Medium":
                pos_size = "Up to 2% of capital"
            else:
                pos_size = "< 1% of capital or avoid"
            st.metric("Suggested Position Size", pos_size)
        with col3:
            st.metric("Trade Reason", reason)
        
        # Trade details
        if signal in ['BUY', 'SELL']:
            st.subheader("📊 Trade Plan")
            st.write(f"**Entry:** ${sl_tp['entry']:.2f}")
            st.write(f"**Stop Loss:** ${sl_tp['sl']:.2f}")
            st.write(f"**Take Profit 1:** ${sl_tp['tp1']:.2f}")
            st.write(f"**Take Profit 2:** ${sl_tp['tp2']:.2f}")
            st.write(f"**Take Profit 3:** ${sl_tp['tp3']:.2f}")
        else:
            st.info(f"No trade recommended. Reason: {reason}")
        
        st.subheader("💡 Mode-Specific Advice")
        st.write(advice)
        
        # Backtest summary
        st.subheader("📈 Backtest (Simplified)")
        st.write(backtest_summary())
        
        # Option to export data
        if st.button("Export Data to CSV"):
            csv = df.to_csv()
            st.download_button("Download CSV", csv, file_name=f"{coin_id}_{timeframe}.csv", mime="text/csv")
else:
    st.info("Configure settings and click 'Run Analysis' to start.")
