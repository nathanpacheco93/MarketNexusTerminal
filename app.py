# app.py — No-auth / No-DB build (works with user_auth shim + local profile_manager)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Modules (no changes needed here; user_auth/profile_manager are the no-DB versions)
from modules import (
    market_data,
    charts,
    portfolio,
    news,
    screener,
    economic_calendar,
    earnings_calendar,
    utils,
    options,
    backtesting,
    risk_management,
    bonds,
    alerts,
    profile_manager,
    futures_overview,
)
from modules.user_auth import user_auth

# ---------- Page config ----------
st.set_page_config(
    page_title="Bloomberg Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive { color: #00FF00; }
    .negative { color: #FF0000; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Session boot ----------
user_auth.initialize_session()

# Title
st.markdown('<h1 class="main-header">🏛️ BLOOMBERG TERMINAL</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
# Optional session info panel (no login, just shows session + reset)
user_auth.display_user_menu()

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Market Overview",
        "Futures Overview",
        "Interactive Charts",
        "Options Chain",
        "Strategy Backtesting",
        "Risk Management",
        "Bond Market",
        "Alerts & Notifications",
        "Portfolio Management",
        "Market Screener",
        "Economic Calendar",
        "Earnings Calendar",
        "Financial News",
        "Watchlist",
        "Profile Management",
    ],
)

# Local defaults (session-only)
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN"]

# (No-DB) load_user_data is a no-op in the shim; harmless to call
if not st.session_state.get("user_data_loaded"):
    user_auth.load_user_data()
    st.session_state.user_data_loaded = True

# ---------- Pages ----------
if page == "Market Overview":
    st.header("📊 Market Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("S&P 500")
        sp500_data = market_data.get_index_data("^GSPC")
        if sp500_data:
            st.metric(
                label="S&P 500",
                value=f"${sp500_data['price']:.2f}",
                delta=f"{sp500_data['change']:.2f} ({sp500_data['change_percent']:.2f}%)",
            )
    with col2:
        st.subheader("NASDAQ")
        nasdaq_data = market_data.get_index_data("^IXIC")
        if nasdaq_data:
            st.metric(
                label="NASDAQ",
                value=f"${nasdaq_data['price']:.2f}",
                delta=f"{nasdaq_data['change']:.2f} ({nasdaq_data['change_percent']:.2f}%)",
            )
    with col3:
        st.subheader("DOW JONES")
        dow_data = market_data.get_index_data("^DJI")
        if dow_data:
            st.metric(
                label="DOW",
                value=f"${dow_data['price']:.2f}",
                delta=f"{dow_data['change']:.2f} ({dow_data['change_percent']:.2f}%)",
            )

    st.header("💱 Currencies & Commodities")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        eur_usd = market_data.get_currency_data("EURUSD=X")
        if eur_usd:
            st.metric("EUR/USD", f"{eur_usd['price']:.4f}", f"{eur_usd['change']:.4f}")
    with col2:
        gbp_usd = market_data.get_currency_data("GBPUSD=X")
        if gbp_usd:
            st.metric("GBP/USD", f"{gbp_usd['price']:.4f}", f"{gbp_usd['change']:.4f}")
    with col3:
        gold = market_data.get_commodity_data("GC=F")
        if gold:
            st.metric("Gold", f"${gold['price']:.2f}", f"{gold['change']:.2f}")
    with col4:
        oil = market_data.get_commodity_data("CL=F")
        if oil:
            st.metric("Crude Oil", f"${oil['price']:.2f}", f"{oil['change']:.2f}")

    st.header("₿ Cryptocurrency")
    col1, col2, col3 = st.columns(3)
    with col1:
        btc = market_data.get_crypto_data("BTC-USD")
        if btc:
            st.metric("Bitcoin", f"${btc['price']:.2f}", f"{btc['change']:.2f}")
    with col2:
        eth = market_data.get_crypto_data("ETH-USD")
        if eth:
            st.metric("Ethereum", f"${eth['price']:.2f}", f"{eth['change']:.2f}")
    with col3:
        ada = market_data.get_crypto_data("ADA-USD")
        if ada:
            st.metric("Cardano", f"${ada['price']:.4f}", f"{ada['change']:.4f}")

elif page == "Futures Overview":
    futures_overview.display_futures_overview()

elif page == "Interactive Charts":
    st.header("📈 Interactive Charts")
    charts.display_charts()

elif page == "Options Chain":
    st.header("📊 Options Chain & Derivatives")
    options.display_options()

elif page == "Strategy Backtesting":
    st.header("🔄 Strategy Backtesting")
    backtesting.display_backtesting()

elif page == "Risk Management":
    st.header("⚠️ Risk Management")
    risk_management.display_risk_management()

elif page == "Bond Market":
    st.header("🏦 Bond Market")
    bonds.display_bonds()

elif page == "Alerts & Notifications":
    st.header("🚨 Alerts & Notifications")
    alerts.display_alerts()

elif page == "Portfolio Management":
    st.header("💼 Portfolio Management")
    portfolio.display_portfolio()

elif page == "Market Screener":
    st.header("🔍 Market Screener")
    screener.display_screener()

elif page == "Economic Calendar":
    st.header("📅 Economic Calendar")
    economic_calendar.display_calendar()

elif page == "Earnings Calendar":
    earnings_calendar.display_earnings_calendar()

elif page == "Financial News":
    st.header("📰 Financial News")
    news.display_news()

elif page == "Watchlist":
    st.header("👁️ Market Watchlist")

    c1, c2 = st.columns([3, 1])
    with c1:
        new_symbol = st.text_input("Add Symbol to Watchlist", placeholder="e.g., AAPL")
    with c2:
        if st.button("Add Symbol"):
            s = (new_symbol or "").strip().upper()
            if s and s not in st.session_state.watchlist:
                st.session_state.watchlist.append(s)
                user_auth.auto_save_watchlist_symbol(s, "add")  # session-only shim
                st.success(f"Added {s} to watchlist")
                st.rerun()

    if st.session_state.watchlist:
        rows = []
        for symbol in list(st.session_state.watchlist):
            try:
                t = yf.Ticker(symbol)
                hist = t.history(period="2d")
                info = getattr(t, "info", {}) or {}
                if not hist.empty and len(hist) >= 2:
                    current_price = float(hist["Close"].iloc[-1])
                    prev_price = float(hist["Close"].iloc[-2])
                    change = current_price - prev_price
                    change_percent = (change / prev_price) * 100 if prev_price else 0.0
                    rows.append(
                        {
                            "Symbol": symbol,
                            "Company": info.get("longName", symbol),
                            "Price": f"${current_price:.2f}",
                            "Change": f"${change:.2f}",
                            "Change %": f"{change_percent:.2f}%",
                            "Volume": f"{info.get('volume', 0):,}" if info.get("volume") else "N/A",
                            "Market Cap": f"${info.get('marketCap', 0)/1e9:.2f}B"
                            if info.get("marketCap")
                            else "N/A",
                        }
                    )
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            ctable, cremove = st.columns([4, 1])
            with ctable:
                st.dataframe(df, use_container_width=True)
            with cremove:
                st.write("Remove:")
                for symbol in list(st.session_state.watchlist):
                    if st.button(f"❌ {symbol}", key=f"remove_{symbol}"):
                        st.session_state.watchlist.remove(symbol)
                        user_auth.auto_save_watchlist_symbol(symbol, "remove")  # session-only shim
                        st.success(f"Removed {symbol}")
                        st.rerun()
        else:
            st.info("No data yet. Try adding another symbol.")
    else:
        st.info("Your watchlist is empty. Add some symbols to get started!")

elif page == "Profile Management":
    profile_manager.display_profile_management()

# No-DB: this is a no-op in the shim, safe to keep for compatibility
user_auth.save_current_state()

# Footer
st.markdown("---")
st.markdown("*Data provided by Yahoo Finance. For educational purposes only.*")
