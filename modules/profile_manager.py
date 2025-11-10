# modules/profile_manager.py — NO-DB / LOCAL-ONLY VERSION

import streamlit as st
import pandas as pd
from datetime import datetime
from modules.user_auth import user_auth  # uses the no-login shim you installed

def display_profile_management():
    """Profile page — session-only, no database."""
    user = user_auth.get_current_user()
    if not user:
        st.error("Profile unavailable.")
        return

    st.header(f"👤 Profile Management — {user['username']}")

    # Summary metrics (from session state)
    portfolio = st.session_state.get("portfolio", [])
    alerts = st.session_state.get("alerts", [])
    watchlist = st.session_state.get("watchlist", [])
    preferences = st.session_state.get("preferences", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Positions", len(portfolio))
    col2.metric("Active Alerts", len(alerts))
    col3.metric("Watchlist Symbols", len(watchlist))
    col4.metric("Saved Preferences", len(preferences))

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "⚙️ Preferences",
        "💾 Export Data",
        "🗑️ Clear Data",
    ])

    with tab1:
        _data_overview(portfolio, alerts, watchlist, preferences)

    with tab2:
        _preferences_editor()

    with tab3:
        _export_data(portfolio, alerts, watchlist, preferences)

    with tab4:
        _clear_data()


def _data_overview(portfolio, alerts, watchlist, preferences):
    st.subheader("📊 Your Data Overview (Session Only)")

    # Portfolio
    st.markdown("### 💼 Portfolio")
    if portfolio:
        df = pd.DataFrame(portfolio)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No portfolio positions.")

    # Alerts
    st.markdown("### 🚨 Alerts")
    if alerts:
        df = pd.DataFrame(alerts)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No alerts.")

    # Watchlist
    st.markdown("### 👁️ Watchlist")
    if watchlist:
        st.code(", ".join(watchlist))
    else:
        st.info("Watchlist is empty.")

    # Preferences
    st.markdown("### ⚙️ Preferences")
    if preferences:
        for (ptype, pkey), val in preferences.items():
            st.write(f"**{ptype} → {pkey}:** {val}")
    else:
        st.info("No saved preferences.")


def _preferences_editor():
    st.subheader("⚙️ Preferences")

    # Auto-save (session-only)
    enabled = st.session_state.get("auto_save_enabled", True)
    new_enabled = st.checkbox("Enable Auto-Save (session only)", enabled)
    if new_enabled != enabled:
        st.session_state.auto_save_enabled = new_enabled
        user_auth.auto_save_preference("system", "auto_save_enabled", new_enabled)
        st.success("Auto-save preference updated.")

    st.markdown("### 📈 Chart Defaults")
    default_type = user_auth.get_user_preference("charts", "default_type", "Candlestick")
    chart_type = st.selectbox("Default Chart Type", ["Candlestick", "Line", "OHLC"],
                              index=["Candlestick", "Line", "OHLC"].index(default_type))

    default_period = user_auth.get_user_preference("charts", "default_period", "1y")
    period = st.selectbox("Default Time Period",
                          ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                          index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"].index(default_period))

    if st.button("Save Chart Preferences"):
        user_auth.auto_save_preference("charts", "default_type", chart_type)
        user_auth.auto_save_preference("charts", "default_period", period)
        st.success("Chart preferences saved.")

    st.markdown("### 🔔 Notification Defaults")
    default_methods = user_auth.get_user_preference("notifications", "default_methods",
                                                    ["Browser Notification"])
    methods = st.multiselect("Default Notification Methods",
                             ["Browser Notification", "Email", "SMS"],
                             default=default_methods)
    if st.button("Save Notification Preferences"):
        user_auth.auto_save_preference("notifications", "default_methods", methods)
        st.success("Notification preferences saved.")


def _export_data(portfolio, alerts, watchlist, preferences):
    st.subheader("💾 Export Data")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("📊 Prepare Portfolio CSV"):
            if portfolio:
                df = pd.DataFrame(portfolio)
                st.download_button(
                    "Download Portfolio CSV",
                    df.to_csv(index=False),
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No portfolio data to export.")

    with c2:
        if st.button("🚨 Prepare Alerts CSV"):
            if alerts:
                df = pd.DataFrame(alerts)
                st.download_button(
                    "Download Alerts CSV",
                    df.to_csv(index=False),
                    file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No alerts data to export.")

    if st.button("👁️ Prepare Watchlist TXT"):
        if watchlist:
            text = "\n".join(watchlist)
            st.download_button(
                "Download Watchlist",
                text,
                file_name="watchlist.txt",
                mime="text/plain",
            )
        else:
            st.warning("No watchlist data to export.")

    if st.button("⚙️ Prepare Preferences JSON"):
        if preferences:
            import json
            data = json.dumps({f"{k[0]}:{k[1]}": v for k, v in preferences.items()}, indent=2)
            st.download_button(
                "Download Preferences JSON",
                data,
                file_name="preferences.json",
                mime="application/json",
            )
        else:
            st.warning("No preferences to export.")


def _clear_data():
    st.subheader("🗑️ Clear Data (Session Only)")
    st.warning("These actions clear data in this Streamlit session only.")

    if st.button("Clear Portfolio"):
        st.session_state.portfolio = []
        st.success("Portfolio cleared.")

    if st.button("Clear Alerts"):
        st.session_state.alerts = []
        st.session_state.alert_history = []
        st.success("Alerts cleared.")

    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []
        st.success("Watchlist cleared.")

    if st.button("Clear Preferences"):
        st.session_state.preferences = {}
        st.success("Preferences cleared.")
