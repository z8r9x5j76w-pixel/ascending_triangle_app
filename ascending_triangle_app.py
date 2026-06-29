"""
Ascending Triangle Scanner — QuantGaps Research
Standalone Streamlit app + self-improver ready
================================================
Run locally:   streamlit run ascending_triangle_app.py
Deploy:        push to GitHub + connect Streamlit Cloud

Same universe as other QuantGaps scanners (tickers from inline list).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ascending Triangle | QuantGaps",
    page_icon="📐",
    layout="wide",
)

# ── Production params (v1.0) ──────────────────────────────────────────────────
SL              = 0.02
TP              = 0.18
MAX_HOLD        = 25
NOTIONAL        = 2000.0
MAX_POSITIONS   = 10
TREND_SMA       = 50
TRADING_DAYS    = 252
PERIOD          = "5y"
BATCH_SIZE      = 20

# Triangle detection params
RESIST_TOL      = 0.04    # resistance touches within 2%
MIN_TOUCHES     = 4       # min touches of resistance
MIN_HIGHER_LOWS = 1       # min rising lows
CONSOL_MIN      = 10      # min consolidation bars
CONSOL_MAX      = 40      # max consolidation bars
POLE_MIN_BARS   = 12      # pole lookback min
POLE_MAX_BARS   = 28      # pole lookback max
POLE_MIN_PCT    = 0.08    # pole min gain
POLE_MAX_PCT    = 0.22    # pole max gain
BRK_BUFFER      = 0.003   # breakout buffer above resistance

# ── Universe ──────────────────────────────────────────────────────────────────
TICKERS = [
    "SPY","QQQ","DIA","IWM","SMH","XLF","XLK","XLV","XLE","XLI",
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","ADBE","CRM",
    "NFLX","AMD","QCOM","TXN","MU","ORCL","CSCO","NOW","AMAT","ISRG",
    "JPM","V","MA","GS","BAC","BLK","UNH","LLY","TMO","ABT",
    "HD","COST","WMT","MCD","NKE","PG","KO","PEP","ABBV","MRK",
    "XOM","CVX","CAT","DE","LMT","RTX","NEE","LIN","MMM","GE",
    "CRWD","PANW","PLTR","SNOW","DDOG","ZS","COIN","SQ","SHOP",
    "SBUX","DIS","PYPL","INTC","IBM","HPQ","F","GM","AAL","DAL",
    "WFC","C","AXP","BRK-B","SCHW","CME","ICE","MCO",
    "AMGN","GILD","BIIB","REGN","VRTX","BSX","MDT","SYK",
    "PLD","AMT","CCI","EQIX","PSA",
]

# ── Data download ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers):
    data = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch in batches:
        try:
            raw = yf.download(batch, period=PERIOD, interval="1d",
                              group_by="ticker", progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            for t in batch:
                try:
                    if hasattr(raw.columns, "levels") and len(raw.columns.levels) > 1:
                        if t not in raw.columns.get_level_values(0):
                            continue
                        df = raw[t].copy()
                    else:
                        df = raw.copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.dropna()
                    if not {"Open","High","Low","Close"}.issubset(df.columns):
                        continue
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.sort_index()
                    if len(df) < 250:
                        continue
                    cols = ["Open","High","Low","Close"]
                    if "Volume" in df.columns:
                        cols.append("Volume")
                    data[t] = df[cols]
                except Exception:
                    continue
        except Exception:
            continue
    return data


# ── Pattern detection ─────────────────────────────────────────────────────────
def detect_ascending_triangle(df, di, p=None):
    """
    Detect ascending triangle at bar di.
    Returns signal dict or None.

    Structure:
      [POLE: strong uptrend] → [CONSOLIDATION: flat top + rising lows] → [BREAKOUT: close above resistance]
    """
    if p is None:
        p = dict(
            RESIST_TOL=RESIST_TOL, MIN_TOUCHES=MIN_TOUCHES,
            MIN_HIGHER_LOWS=MIN_HIGHER_LOWS,
            CONSOL_MIN=CONSOL_MIN, CONSOL_MAX=CONSOL_MAX,
            POLE_MIN_BARS=POLE_MIN_BARS, POLE_MAX_BARS=POLE_MAX_BARS,
            POLE_MIN_PCT=POLE_MIN_PCT, POLE_MAX_PCT=POLE_MAX_PCT,
            BRK_BUFFER=BRK_BUFFER, TREND_SMA=TREND_SMA,
        )

    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values

    min_start = p["POLE_MAX_BARS"] + p["CONSOL_MAX"] + p["TREND_SMA"] + 5
    if di < min_start or di >= len(close) - 1:
        return None

    # SMA trend filter
    sma = np.mean(close[di - p["TREND_SMA"]:di])
    if close[di] <= sma:
        return None

    # Previous bar must be below or at resistance (not already broken out)
    # We'll establish resistance from the consolidation window

    for consol_bars in range(p["CONSOL_MIN"], p["CONSOL_MAX"] + 1):
        consol_start = di - consol_bars
        if consol_start < p["POLE_MAX_BARS"] + 2:
            continue

        consol_highs = high[consol_start:di]   # highs during consolidation
        consol_lows  = low[consol_start:di]    # lows during consolidation
        consol_close = close[consol_start:di]

        if len(consol_highs) < p["CONSOL_MIN"]:
            continue

        # ── Flat resistance: find the resistance level ────────────────────
        resist_level = float(np.max(consol_highs))

        # Count touches: bars where high is within RESIST_TOL of resist_level
        touches = int(np.sum(
            np.abs(consol_highs - resist_level) / resist_level <= p["RESIST_TOL"]
        ))
        if touches < p["MIN_TOUCHES"]:
            continue

        # No close above resistance during consolidation
        if np.any(consol_close > resist_level * (1 + p["RESIST_TOL"])):
            continue

        # ── Rising lows: check that lows trend upward ─────────────────────
        # Divide consolidation into thirds and compare low of each third
        third = max(1, consol_bars // 3)
        lows_early  = float(np.min(consol_lows[:third]))
        lows_mid    = float(np.min(consol_lows[third:2*third]))
        lows_late   = float(np.min(consol_lows[2*third:]))

        higher_low_count = 0
        if lows_mid > lows_early * (1 + 0.005):
            higher_low_count += 1
        if lows_late > lows_mid * (1 + 0.005):
            higher_low_count += 1
        if lows_late > lows_early * (1 + 0.005):
            higher_low_count += 1

        if higher_low_count < p["MIN_HIGHER_LOWS"]:
            continue

        # ── Breakout check ────────────────────────────────────────────────
        # Previous bar below resistance, current bar above
        if close[di - 1] > resist_level:
            continue
        if close[di] < resist_level * (1 + p["BRK_BUFFER"]):
            continue

        # ── Pole: strong uptrend before consolidation ─────────────────────
        pole_found = False
        for pole_bars in range(p["POLE_MIN_BARS"], p["POLE_MAX_BARS"] + 1):
            pole_start = consol_start - pole_bars
            if pole_start < 1:
                continue
            pole_low  = float(np.min(low[pole_start:consol_start]))
            pole_high = float(np.max(high[pole_start:consol_start]))
            pole_gain = (pole_high - pole_low) / pole_low
            if p["POLE_MIN_PCT"] <= pole_gain <= p["POLE_MAX_PCT"]:
                pole_found = True
                break

        if not pole_found:
            continue

        # ── Signal found ──────────────────────────────────────────────────
        triangle_width = (resist_level - float(np.min(consol_lows))) / resist_level
        return {
            "resist_level":   round(resist_level, 2),
            "touches":        touches,
            "higher_lows":    higher_low_count,
            "consol_bars":    consol_bars,
            "pole_gain_pct":  round(pole_gain * 100, 1),
            "triangle_width": round(triangle_width * 100, 1),
            "strength":       round(touches * 0.4 + higher_low_count * 0.3 + pole_gain * 0.3, 3),
        }

    return None


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(data, p=None):
    date_set = set()
    for df in data.values():
        date_set.update(df.index.tolist())
    dates = sorted(date_set)

    signals = {}
    for ticker, df in data.items():
        tsigs = {}
        for di in range(1, len(df)):
            date = df.index[di]
            if date not in date_set:
                continue
            sig = detect_ascending_triangle(df, di, p)
            if not sig:
                continue
            bl = sig["resist_level"]
            ct = float(df["Close"].iloc[di])
            cy = float(df["Close"].iloc[di - 1])
            if not (cy <= bl < ct):
                continue
            tsigs[date] = sig
        if tsigs:
            signals[ticker] = tsigs

    open_pos, pending, trades = {}, {}, []
    daily_pnl = np.zeros(len(dates))

    for di in range(1, len(dates)):
        date = dates[di]
        if date in pending:
            for ticker, strength, signal in sorted(pending.pop(date),
                                                   key=lambda x: x[1], reverse=True):
                if len(open_pos) >= MAX_POSITIONS or ticker in open_pos:
                    continue
                df = data.get(ticker)
                if df is None or date not in df.index:
                    continue
                o = float(df.loc[date, "Open"])
                if not np.isfinite(o) or o <= 0:
                    continue
                if o < signal["resist_level"]:
                    continue
                sl  = p["SL"]  if p else SL
                tp  = p["TP"]  if p else TP
                mh  = p["MAX_HOLD"] if p else MAX_HOLD
                open_pos[ticker] = dict(
                    entry_price=o, shares=NOTIONAL/o,
                    sl_price=o*(1-sl), tp_price=o*(1+tp),
                    days_held=0, entry_di=di, entry_date=date,
                    resist=signal["resist_level"],
                    sl=sl, tp=tp, mh=mh,
                )

        day_pnl, to_close = 0.0, []
        for ticker, pos in open_pos.items():
            df = data.get(ticker)
            if df is None or date not in df.index:
                continue
            bar = df.loc[date]
            lo = float(bar["Low"]); hi = float(bar["High"]); cl = float(bar["Close"])
            pos["days_held"] += 1
            reason = ep = None
            if   np.isfinite(lo) and lo <= pos["sl_price"]: reason, ep = "SL", pos["sl_price"]
            elif np.isfinite(hi) and hi >= pos["tp_price"]: reason, ep = "TP", pos["tp_price"]
            elif pos["days_held"] >= pos["mh"]:             reason, ep = "MH", cl
            if reason:
                pnl = (ep - pos["entry_price"]) * pos["shares"]
                day_pnl += pnl
                trades.append({
                    "ticker":       ticker,
                    "entry_date":   pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date":    date.strftime("%Y-%m-%d"),
                    "entry_price":  round(pos["entry_price"], 2),
                    "exit_price":   round(ep, 2),
                    "pnl":          round(pnl, 2),
                    "reason":       reason,
                    "days_held":    pos["days_held"],
                })
                to_close.append(ticker)

        daily_pnl[di] = day_pnl
        for t in to_close:
            open_pos.pop(t, None)

        if di < len(dates) - 1:
            next_date = dates[di+1]
            for ticker, tsigs in signals.items():
                if ticker in open_pos or date not in tsigs:
                    continue
                sig = tsigs[date]
                pending.setdefault(next_date, []).append(
                    (ticker, sig["strength"], sig))

    return trades, daily_pnl, len(dates)


def calc_metrics(trades, daily_pnl, n_dates):
    if not trades:
        return {}
    pnls    = np.array([t["pnl"] for t in trades])
    reasons = [t["reason"] for t in trades]
    holds   = np.array([t["days_held"] for t in trades], dtype=float)
    n       = len(trades)
    capital = NOTIONAL * MAX_POSITIONS
    n_years = n_dates / TRADING_DAYS
    total   = float(pnls.sum())
    cagr    = ((1 + total/capital)**(1/n_years) - 1)*100 if n_years else 0
    cum     = np.cumsum(daily_pnl)
    std     = daily_pnl.std()
    sharpe  = daily_pnl.mean()/std*np.sqrt(TRADING_DAYS) if std > 0 else 0
    peak    = np.maximum.accumulate(cum)
    max_dd  = float((cum - peak).min())
    calmar  = cagr/abs(max_dd/capital*100) if max_dd else 0
    return dict(
        n=n, wr=round((pnls>0).sum()/n*100,1), total=round(total,2),
        cagr=round(cagr,2), sharpe=round(sharpe,3),
        calmar=round(calmar,3), max_dd=round(max_dd,2),
        avg_hold=round(float(holds.mean()),1),
        pct_sl=round(reasons.count("SL")/n*100,1),
        pct_tp=round(reasons.count("TP")/n*100,1),
        pct_mh=round(reasons.count("MH")/n*100,1),
        cum_pnl=cum,
    )


# ── Live signal scan ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def find_live_signals(_data):
    cutoff  = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(5)
    results = []
    for ticker, df in _data.items():
        for di in range(max(1, len(df) - 10), len(df)):
            if df.index[di] < cutoff:
                continue
            sig = detect_ascending_triangle(df, di)
            if not sig:
                continue
            bl = sig["resist_level"]
            ct = float(df["Close"].iloc[di])
            cy = float(df["Close"].iloc[di - 1])
            if not (cy <= bl < ct):
                continue
            results.append({
                "Ticker":         ticker,
                "Date":           df.index[di].strftime("%Y-%m-%d"),
                "Close":          round(ct, 2),
                "Resistance":     round(bl, 2),
                "SL Price":       round(ct * (1 - SL), 2),
                "TP Price":       round(ct * (1 + TP), 2),
                "Touches":        sig["touches"],
                "Higher Lows":    sig["higher_lows"],
                "Consol Bars":    sig["consol_bars"],
                "Pole Gain %":    sig["pole_gain_pct"],
                "Strength":       sig["strength"],
            })
    return sorted(results, key=lambda x: x["Strength"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("📐 Ascending Triangle Scanner")
st.caption("QuantGaps Research · v1.0 · SL 2% · TP 12% · MaxHold 20 · SMA50")

tab1, tab2 = st.tabs(["🟢 LIVE SIGNALS", "🔵 BACKTEST"])

with tab1:
    st.subheader("Live Ascending Triangle Breakouts — last 5 trading days")
    st.caption(f"Universe: {len(TICKERS)} tickers · Entry: next-day open > resistance level")

    if st.button("▶ Run Live Scan", type="primary", key="live"):
        with st.spinner("Downloading data..."):
            data = download_data(tuple(TICKERS))
        with st.spinner(f"Scanning {len(data)} tickers..."):
            sigs = find_live_signals(data)

        if not sigs:
            st.info("No ascending triangle breakouts in the last 5 trading days.")
        else:
            st.success(f"✅ {len(sigs)} signal(s) found")
            st.dataframe(pd.DataFrame(sigs), use_container_width=True, hide_index=True)
            st.caption("⚠️ Entry next trading day at open, only if open > Resistance level")

            top = sigs[0]
            st.divider()
            st.markdown(f"### 🏆 Top Signal: **{top['Ticker']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Close",      f"${top['Close']}")
            c2.metric("Resistance", f"${top['Resistance']}")
            c3.metric("SL",  f"${top['SL Price']}", f"-{SL*100:.0f}%", delta_color="inverse")
            c4.metric("TP",  f"${top['TP Price']}", f"+{TP*100:.0f}%")

with tab2:
    st.subheader("5-Year Backtest · Ascending Triangle v1.0")
    st.caption("Reference only — historical simulation, not forward-looking")

    if st.button("▶ Run Backtest", type="primary", key="bt"):
        with st.spinner("Downloading 5y data..."):
            data = download_data(tuple(TICKERS))
        with st.spinner("Running backtest..."):
            trades, daily_pnl, n_dates = run_backtest(data)
            m = calc_metrics(trades, daily_pnl, n_dates)

        if not m:
            st.warning("No trades generated — try relaxing detection parameters.")
        else:
            st.divider()
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Trades",    m["n"])
            c2.metric("Win Rate",  f"{m['wr']}%")
            c3.metric("CAGR",      f"{m['cagr']}%")
            c4.metric("Sharpe",    m["sharpe"])
            c5.metric("Calmar",    m["calmar"])
            c6.metric("Max DD",    f"${m['max_dd']:,.0f}")

            c1b,c2b,c3b,c4b = st.columns(4)
            c1b.metric("Total P&L", f"${m['total']:,.0f}")
            c2b.metric("Avg Hold",  f"{m['avg_hold']} days")
            c3b.metric("SL exits",  f"{m['pct_sl']}%")
            c4b.metric("TP exits",  f"{m['pct_tp']}%")

            st.divider()
            st.markdown("#### Equity Curve")
            st.line_chart(pd.DataFrame({"Cumulative P&L ($)": m["cum_pnl"]}),
                          use_container_width=True)

            st.divider()
            st.markdown("#### Trade Log")
            df_t = pd.DataFrame(trades).sort_values("exit_date", ascending=False)
            st.dataframe(df_t.astype(str), use_container_width=True, hide_index=True)

            csv = df_t.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv,
                               "ascending_triangle_trades.csv", "text/csv")
