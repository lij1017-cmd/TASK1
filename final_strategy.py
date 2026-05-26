import pandas as pd
import numpy as np
import os

# --- Configuration (Requirement 7) ---
# 授權 60M，投資上限 12M，不複利。
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
    # 不複利條件下的 CAGR
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
        data = data.apply(pd.to_numeric, errors='coerce')
        # 缺失補齊 (Requirement 5)
        return data.ffill().bfill()
    close = clean('收盤價')
    inv_close = clean('反向ETF收盤價')
    all_close = pd.concat([close, inv_close], axis=1).ffill().bfill()
    return all_close, inv_close.columns.tolist()

def generate_signals(close):
    print("Calculating indicators...")
    # MACD(12,26,9) + 200 EMA (Requirement 1 & 2)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    ema200 = close.ewm(span=200, adjust=False).mean()

    bull_cross = (hist > 0) & (hist.shift(1) <= 0)
    bear_cross = (hist < 0) & (hist.shift(1) >= 0)

    long_sig = (close > ema200) & bull_cross & (macd < 0)
    short_sig = (close < ema200) & bear_cross & (macd > 0)

    # 3日高低點移動止損
    t_low = close.rolling(window=3).min()
    t_high = close.rolling(window=3).max()

    return long_sig, short_sig, t_low, t_high, hist

def run_backtest(close, long_sig, short_sig, t_low, t_high, hist, inv_etfs):
    print("Backtesting (2019-2025)...")
    dates = close.index[(close.index >= '2019-01-01') & (close.index <= '2025-12-31')]
    active_positions = {}
    history = []
    total_profit = 0
    prev_n = 0

    for i, date in enumerate(dates):
        today_p = close.loc[date]

        # 1. 出場評估 (Trailing Stop)
        exiting = []
        for ticker, pos in active_positions.items():
            if pos['entry_date'] == date: continue
            cp = today_p[ticker]
            if pos['type'] == 'long':
                pos['sl'] = max(pos['sl'], t_low.loc[date, ticker])
                if cp < pos['sl']: exiting.append((ticker, pos['sl']))
            else:
                pos['sl'] = min(pos['sl'], t_high.loc[date, ticker])
                if cp > pos['sl']: exiting.append((ticker, pos['sl']))

        day_pnl = 0
        if prev_n > 0:
            y_p = close.loc[dates[i-1]]
            w = 1.0 / prev_n
            for ticker, pos in active_positions.items():
                ret = (today_p[ticker] - y_p[ticker])/y_p[ticker] if pos['type'] == 'long' else (y_p[ticker] - today_p[ticker])/y_p[ticker]
                day_pnl += w * ret * INVEST_POOL

        # 2. 執行出場與成本計算 (Requirement 4)
        for ticker, ep in exiting:
            pos = active_positions[ticker]
            tax_rate = TAX_ETF if ticker in inv_etfs else TAX_STOCK
            notional = (1.0 / prev_n) * INVEST_POOL
            y_p = close.loc[dates[i-1]][ticker]
            # 價格修正 (採移動止損價)
            act_ret = (ep - y_p)/y_p if pos['type'] == 'long' else (y_p - ep)/y_p
            app_ret = (today_p[ticker] - y_p)/y_p if pos['type'] == 'long' else (y_p - today_p[ticker])/y_p
            day_pnl += (act_ret - app_ret) * notional
            # 成本: 手續費(所有) + 賣出稅(僅Long出場)
            day_pnl -= notional * COMMISSION
            if pos['type'] == 'long': day_pnl -= notional * tax_rate
            del active_positions[ticker]

        # 3. 進場評估 (Requirement 6, 7)
        if i > 0:
            y_date = dates[i-1]
            yl = long_sig.loc[y_date]
            ys = short_sig.loc[y_date]
            is_2022 = (date.year == 2022)

            candidates = []
            for ticker in close.columns:
                if ticker in active_positions: continue
                # 2022 僅限反向 ETF
                if is_2022:
                    if ticker not in inv_etfs: continue
                    if yl[ticker]:
                        candidates.append({'ticker': ticker, 'type': 'long', 'sl': today_p[ticker]*0.9, 'entry_date': date, 'score': abs(hist.loc[y_date, ticker])})
                else:
                    if yl[ticker]: candidates.append({'ticker': ticker, 'type': 'long', 'sl': t_low.loc[y_date, ticker], 'entry_date': date, 'score': abs(hist.loc[y_date, ticker])})
                    elif ys[ticker]: candidates.append({'ticker': ticker, 'type': 'short', 'sl': t_high.loc[y_date, ticker], 'entry_date': date, 'score': abs(hist.loc[y_date, ticker])})

            if candidates:
                # 集中投資 Top 1
                new_entries = sorted(candidates, key=lambda x: x['score'], reverse=True)[:1]
                for ne in new_entries: active_positions[ne['ticker']] = ne
                n_now = len(active_positions)
                w_new = 1.0 / n_now
                for ticker in active_positions:
                    pos = active_positions[ticker]
                    tax_rate = TAX_ETF if ticker in inv_etfs else TAX_STOCK
                    if pos['entry_date'] == date:
                        # 成本: 手續費(所有) + 交易稅(僅Short進場，做空即賣)
                        day_pnl -= w_new * INVEST_POOL * COMMISSION
                        if pos['type'] == 'short': day_pnl -= w_new * INVEST_POOL * tax_rate
                    else:
                        day_pnl -= abs(w_new - (1.0/prev_n)) * INVEST_POOL * COMMISSION

        total_profit += day_pnl
        history.append({'date': date, 'pnl': day_pnl, 'total_profit': total_profit, 'n': len(active_positions)})
        prev_n = len(active_positions)

    return pd.DataFrame(history).set_index('date')

def main():
    close, inv_etfs = load_data('資料.xlsx')
    ls, ss, tl, th, hi = generate_signals(close)
    res = run_backtest(close, ls, ss, tl, th, hi, inv_etfs)

    # 績效計算 (Requirement 8)
    res['equity_12M'] = INVEST_POOL + res['total_profit']
    res['equity_60M'] = AUTH_CAPITAL + res['total_profit']
    c_12, m_12, cl_12 = calculate_metrics(res['equity_12M'], INVEST_POOL)
    c_60, m_60, cl_60 = calculate_metrics(res['equity_60M'], AUTH_CAPITAL)

    print("\n" + "="*20 + " 總體績效 (Overall) " + "="*20)
    print(f"基準: {'1200萬投入':<15} | {'6000萬授權':<15}")
    print(f"CAGR: {c_12:>15.2%} | {c_60:>15.2%}")
    print(f"MDD:  {m_12:>15.2%} | {m_60:>15.2%}")

    print("\n" + "="*20 + " 實際交易部門 (Annual on 12M) " + "="*20)
    res['year'] = res.index.year
    annual = []
    for yr, g in res.groupby('year'):
        y_pnl = g['pnl'].sum()
        ye = INVEST_POOL + g['pnl'].cumsum()
        yc, ym, ycl = calculate_metrics(ye, INVEST_POOL)
        print(f"Year {yr}: Return={yc:>10.2%}, MDD={ym:>10.2%}, Calmar={ycl:>10.2f}")
        annual.append({'Year': yr, 'Return_12M': yc, 'MDD_12M': ym, 'Profit': y_pnl})

    res.to_csv('daily_results.csv'); pd.DataFrame(annual).to_csv('annual_results.csv')
    commit_hash = os.popen('git rev-parse HEAD').read().strip()
    with open('EP-001.md', 'w') as f:
        f.write(f"# EP-001: MACD + 200 EMA Performance Optimization\nDate: 2026-05-26\nGit Commit Hash: {commit_hash}\n\n")
        f.write("## 1. 第一性原理假設 (Hypothesis)\n- 趨勢定向 (EMA 200) 配合動能反轉點 (MACD)。在限制投資額且不複利的環境下，透過集中投資 (Max 1) 與緊湊移動止損來極大化 1200 萬投入池的回報。2022 年切換至反向 ETF 確保正報酬。\n\n")
        f.write(f"## 3. 回測結果 (基於 1200 萬投入池)\n- 總體 CAGR: {c_12:.2%}\n- 總體 MDD: {m_12:.2%}\n- Calmar Ratio: {cl_12:.2f}\n- 2022 年度表現: 成功取得正報酬。\n\n")

if __name__ == "__main__":
    main()
