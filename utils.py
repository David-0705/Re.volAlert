# utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import ta

# -------------------------
# Sequence & Feature Utils
# -------------------------

SEQ_LEN = 30  # or whatever value you used during model training

FEATURE_COLS = ['Return', 'MA_5', 'MA_10', 'Volatility_10', 'Momentum', 'MA_Ratio']

def create_sequences_multi(df, seq_len=30):
    X_seq, y_class, y_reg, stock_ids = [], [], [], []
    le = LabelEncoder()
    df['Ticker_ID'] = le.fit_transform(df['Ticker'])
    
    for ticker, subdf in df.groupby('Ticker'):
        arr = subdf[FEATURE_COLS].values
        cls = subdf['Target_Class'].values if 'Target_Class' in subdf.columns else np.zeros(len(subdf))
        reg = subdf['Target_Reg'].values if 'Target_Reg' in subdf.columns else np.zeros(len(subdf))
        tid = subdf['Ticker_ID'].iloc[0]
        for i in range(len(subdf) - seq_len):
            X_seq.append(arr[i:i+seq_len])
            y_class.append(cls[i+seq_len])
            y_reg.append(reg[i+seq_len])
            stock_ids.append(tid)
    return np.array(X_seq), np.array(y_class), np.array(y_reg), np.array(stock_ids), le

def feature_engineering(df):
    df['Return'] = df['Adj Close'].pct_change()
    df['MA_5'] = df['Adj Close'].rolling(5).mean()
    df['MA_10'] = df['Adj Close'].rolling(10).mean()
    df['Volatility_10'] = df['Return'].rolling(10).std()
    df['Momentum'] = df['Adj Close'] / df['Adj Close'].shift(5) - 1
    df['MA_Ratio'] = df['MA_5'] / df['MA_10'] - 1
    df = df.dropna()
    df['Target_Class'] = (df['Return'].shift(-1) > 0).astype(int)
    df['Target_Reg'] = df['Return'].shift(-1)
    return df.dropna()

# -------------------------
# Prediction Utils
# -------------------------
def prepare_test_df(ticker, df_feat, model, scaler, le, seq_len=30):
    df_sub = df_feat[df_feat['Ticker'] == ticker].copy().reset_index(drop=True)
    tid = le.transform([ticker])[0]
    
    X_pred, y_class, y_reg, _, _ = create_sequences_multi(df_sub)
    X_pred = scaler.transform(X_pred.reshape(-1, len(FEATURE_COLS))).reshape(-1, seq_len, len(FEATURE_COLS))
    stock_array = np.full((len(X_pred), 1), tid)
    
    preds_class, preds_reg = model.predict([X_pred, stock_array], verbose=0)
    preds_class = (preds_class.flatten() > 0.5).astype(int)
    
    df_test = df_sub.iloc[seq_len:].copy()
    df_test['pred'] = preds_class
    df_test['pred_prob'] = preds_class
    df_test['next_ret'] = df_test['Target_Reg']
    
    return df_test

def predict_for_stock(ticker, df_feat, model, scaler, le, seq_len=30):
    df_test = prepare_test_df(ticker, df_feat, model, scaler, le, seq_len)
    last_pred_dir = 'UP' if df_test['pred'].iloc[-1] == 1 else 'DOWN'
    last_pred_ret = df_test['next_ret'].iloc[-1] * 100
    print(f'📈 {ticker} Prediction: {last_pred_dir}, Expected Return: {last_pred_ret:.2f}%')
    return df_test

# -------------------------
# Backtesting & Plotting
# -------------------------
def simulate_strategy(df, initial_capital=100000.0, transaction_cost=0.0005):
    df = df.copy().reset_index(drop=False)
    n = len(df)
    equity = np.zeros(n)
    equity[0] = initial_capital
    prev_pos = 0
    daily_ret_strategy = []
    equity_series = []
    
    for i in range(n):
        pred = int(df.loc[i, 'pred'])
        r = float(df.loc[i, 'next_ret'])
        tx_cost = 0.0
        if pred != prev_pos:
            tx_cost = transaction_cost * equity[i-1] if i>0 else transaction_cost * initial_capital
        prev_equity = equity[i-1] if i>0 else initial_capital
        new_equity = prev_equity * (1 + r) - tx_cost if pred==1 else prev_equity - tx_cost
        equity[i] = new_equity
        daily_ret_strategy.append((new_equity-prev_equity)/prev_equity)
        equity_series.append(new_equity)
        prev_pos = pred
    
    df['strategy_equity'] = equity_series
    df['strategy_daily_ret'] = daily_ret_strategy
    bh_eq = initial_capital * np.cumprod(1 + df['next_ret'].values)
    df['bh_equity'] = bh_eq
    df['bh_daily_ret'] = df['next_ret'].values

    def calc_metrics(equity_series, daily_returns):
        total_return = equity_series[-1]/equity_series[0]-1
        days = len(equity_series)
        annual_factor = 252
        cagr = (equity_series[-1]/equity_series[0])**(annual_factor/days)-1
        ann_vol = np.std(daily_returns)*np.sqrt(annual_factor)
        ann_return = np.mean(daily_returns)*annual_factor
        sharpe = ann_return/(ann_vol+1e-12)
        running_max = np.maximum.accumulate(equity_series)
        max_dd = ((equity_series - running_max)/running_max).min()
        return {
            'total_return': total_return,
            'CAGR': cagr,
            'annualized_return': ann_return,
            'annualized_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }
    
    stats = {
        'initial_capital': initial_capital,
        'n_days': n,
        'n_trades': int(df[df['pred']==1].shape[0]),
        'win_rate': (df[(df['pred']==1)&(df['next_ret']>0)].shape[0])/(df[df['pred']==1].shape[0]+1e-12),
        'avg_ret_per_trade': df[df['pred']==1]['next_ret'].mean() if df[df['pred']==1].shape[0]>0 else 0,
        'strategy': calc_metrics(df['strategy_equity'].values, df['strategy_daily_ret'].values),
        'buy_and_hold': calc_metrics(df['bh_equity'].values, df['bh_daily_ret'].values)
    }
    return df.set_index('Date' if 'Date' in df.columns else df.index), stats

def plot_performance(df, stats):
    fig, axes = plt.subplots(2,2, figsize=(14,8), gridspec_kw={'height_ratios':[2,1]})
    ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    ax1.plot(df.index, df['strategy_equity'], label='Strategy Equity', linewidth=2)
    ax1.plot(df.index, df['bh_equity'], label='Buy & Hold Equity', linewidth=2)
    ax1.set_title("Equity Curve: Strategy vs Buy & Hold")
    ax1.legend()
    running_max = np.maximum.accumulate(df['strategy_equity'].values)
    drawdown = (df['strategy_equity'].values - running_max)/running_max
    ax2.plot(df.index, drawdown, color='red')
    ax2.fill_between(df.index, drawdown, 0, color='red', alpha=0.2)
    ax2.set_title(f"Strategy Drawdown (max dd = {stats['strategy']['max_drawdown']:.2%})")
    ax3.hist(df['strategy_daily_ret'].dropna(), bins=40)
    ax3.set_title("Strategy daily returns distribution")
    if 'pred_prob' in df.columns:
        ax4.scatter(df['pred_prob'], df['next_ret'], alpha=0.6, s=20)
        ax4.set_xlabel("Predicted probability")
        ax4.set_ylabel("Actual next-day return")
        ax4.set_title("Predicted prob vs Actual next-day return")
    plt.tight_layout()
    plt.show()
