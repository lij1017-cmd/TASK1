import pandas as pd
import numpy as np
import os

# --- Configuration (Requirement 7) ---
# 總授權金額 6000 萬，每次可用資金上限 1200 萬
AUTH_CAPITAL = 60000000
INVEST_POOL = 12000000
COMMISSION = 0.001425
TAX_STOCK = 0.003
TAX_ETF = 0.001

def calculate_metrics(equity_ser, initial_cap):
    if len(equity_ser) < 2: return 0, -1, 0
    total_profit = equity_ser.iloc[-1] - initial_cap
    total_ret = total_profit / initial_cap
    days = (equity_ser.index[-1] - equity_ser.index[0]).days
    years = max(days / 365.25, 0.1)
    cagr = (1 + total_ret)**(1/years) - 1 if (1+total_ret) > 0 else -1
    peak = equity_ser.cummax()
    dd = (equity_ser - peak) / peak
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return cagr, mdd, calmar

def load_data(path):
    print("Loading data...")
    xls = pd.ExcelFile(path)
    def clean(sheet):
        df = pd.read_excel(xls, sheet)
        data = df.iloc[1:].copy()
        data['date'] = pd.to_datetime(data.iloc[:, 0].str.extract(r'(\d{8})')[0], format='%Y%m%d')
        data = data.drop(columns=[data.columns[0]]).set_index('date')
        return data.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    close = clean('收盤價')
    inv_close = clean('反向ETF收盤價')
    all_close = pd.concat([close, inv_close], axis=1).ffill().bfill()
    return all_close, inv_close.columns.tolist()

def generate_signals(close, fast=12, slow=26, signal=9, ema_len=200):
    print(f"Calculating indicators (MACD={fast,slow,signal}, EMA={ema_len})...")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    ema_f = close.ewm(span=ema_len, adjust=False).mean()
    bull_cross = (hist > 0) & (hist.shift(1) <= 0)
    bear_cross = (hist < 0) & (hist.shift(1) >= 0)

    # 策略核心邏輯 (Triple Filter)
    # 1. 趨勢定位: 價格位於 200 EMA 之上 (多) 或之下 (空)
    # 2. 位階篩選: MACD Line < 0 (多) 或 > 0 (空)，確保進場在超跌/超漲點
    # 3. 動能確認: MACD 柱狀體黃金交叉 (多) 或死亡交叉 (空)
    long_sig = (close > ema_f) & (macd < 0) & bull_cross
    short_sig = (close < ema_f) & (macd > 0) & bear_cross

    return long_sig, short_sig, hist

def run_backtest(close, long_sig, short_sig, hist, inv_etfs, mp=5):
    print(f"Starting backtest simulation (Max Holdings: {mp})...")
    dates = close.index[(close.index >= '2019-01-01') & (close.index <= '2025-12-31')]
    active_positions = {}
    history = []
    trades = []
    total_profit = 0
    prev_n = 0

    for i, date in enumerate(dates):
        today_p = close.loc[date]

        # 1. 出場評估 (Exit Check)
        exiting = []
        for ticker, pos in active_positions.items():
            if pos['entry_date'] == date: continue
            cp = today_p[ticker]
            if isinstance(cp, pd.Series): cp = cp.iloc[0]

            # Trailing Stop: 1% 緊湊移動止損
            if cp < pos['sl']:
                exiting.append((ticker, pos['sl'], "StopLoss"))
            else:
                pos['sl'] = max(pos['sl'], cp * 0.99)

        # 每日盈虧計算 (PnL Calculation)
        day_pnl = 0
        if prev_n > 0:
            y_p = close.loc[dates[i-1]]
            w = 1.0 / prev_n
            for ticker, pos in active_positions.items():
                ypv = y_p[ticker]; tpv = today_p[ticker]
                if isinstance(ypv, pd.Series): ypv = ypv.iloc[0]
                if isinstance(tpv, pd.Series): tpv = tpv.iloc[0]
                day_pnl += w * ((tpv - ypv)/ypv) * INVEST_POOL

        # 執行出場 (Execute Exits)
        for ticker, ep, reason in exiting:
            pos = active_positions[ticker]
            tax_rate = TAX_ETF if ticker in inv_etfs else TAX_STOCK
            notional = (1.0 / prev_n) * INVEST_POOL
            y_p_val = close.loc[dates[i-1]][ticker]
            if isinstance(y_p_val, pd.Series): y_p_val = y_p_val.iloc[0]
            tp_val = today_p[ticker]
            if isinstance(tp_val, pd.Series): tp_val = tp_val.iloc[0]

            # 調整為實際成交價與今日收盤價的差異
            act_ret = (ep - y_p_val)/y_p_val
            app_ret = (tp_val - y_p_val)/y_p_val
            day_pnl += (act_ret - app_ret) * notional

            # 交易成本 (Requirement 4)
            cost = notional * (COMMISSION + tax_rate)
            day_pnl -= cost

            trades.append({
                'ticker': ticker,
                'type': pos['type'],
                'entry_date': pos['entry_date'],
                'entry_price': pos['entry_p'],
                'exit_date': date,
                'exit_price': ep,
                'reason': reason,
                'pnl_net': (act_ret * notional) - cost
            })
            del active_positions[ticker]

        # 2. 進場評估 (Entry Evaluation)
        if i > 0:
            y_date = dates[i-1]
            yl = long_sig.loc[y_date]
            is_2022 = (date.year == 2022)

            candidates = []
            for ticker in close.columns:
                if ticker in active_positions: continue
                tpv = today_p[ticker]
                if isinstance(tpv, pd.Series): tpv = tpv.iloc[0]
                ypv = close.loc[y_date, ticker]
                if isinstance(ypv, pd.Series): ypv = ypv.iloc[0]

                if is_2022:
                    if ticker in inv_etfs and yl[ticker]:
                        candidates.append({'ticker': ticker, 'type': 'long', 'sl': tpv*0.99, 'entry_p': tpv, 'score': abs(hist.loc[y_date, ticker])})
                else:
                    if yl[ticker]:
                        candidates.append({'ticker': ticker, 'type': 'long', 'sl': tpv*0.99, 'entry_p': tpv, 'score': abs(hist.loc[y_date, ticker])})

            if candidates:
                space = mp - len(active_positions)
                for ne in sorted(candidates, key=lambda x: x['score'], reverse=True)[:max(space, 0)]:
                    ne['entry_date'] = date
                    active_positions[ne['ticker']] = ne

                n_now = len(active_positions)
                if n_now > 0:
                    w_new = 1.0 / n_now
                    for ticker, pos in active_positions.items():
                        if pos['entry_date'] == date:
                            day_pnl -= w_new * INVEST_POOL * COMMISSION
                        else:
                            day_pnl -= abs(w_new - (1.0/prev_n)) * INVEST_POOL * COMMISSION

        total_profit += day_pnl
        history.append({'date': date, 'pnl': day_pnl, 'total_profit': total_profit, 'n': len(active_positions)})
        prev_n = len(active_positions)

    return pd.DataFrame(history).set_index('date'), pd.DataFrame(trades)

def main():
    close, inv_etfs = load_data('資料.xlsx')
    p = {'fast': 12, 'slow': 26, 'signal': 9, 'ema_len': 200}
    res, trades = run_backtest(close, *generate_signals(close, **p), inv_etfs, mp=5)

    res['equity_12M'] = INVEST_POOL + res['total_profit']
    c_12, m_12, cl_12 = calculate_metrics(res['equity_12M'], INVEST_POOL)

    print("\n" + "="*40)
    print(f"Overall CAGR (12M Base): {c_12:.2%}")
    print(f"Overall MDD (12M Base): {m_12:.2%}")
    print(f"Calmar Ratio: {cl_12:.2f}")

    # Save Outputs
    res.to_csv('daily_results.csv')
    trades.to_csv('trade_log.csv')

    # Annual Summary
    res['year'] = res.index.year
    annual = []
    for yr, g in res.groupby('year'):
        ye = INVEST_POOL + g['pnl'].cumsum()
        yc, ym, ycl = calculate_metrics(ye, INVEST_POOL)
        annual.append({'Year': yr, 'Return_12M': yc, 'MDD_12M': ym, 'Profit': g['pnl'].sum()})
    pd.DataFrame(annual).to_csv('annual_results.csv')

if __name__ == "__main__":
    main()
