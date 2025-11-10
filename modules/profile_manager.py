import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from modules.database import db_manager
from modules.user_auth import user_auth

def display_profile_management():
    """Display user profile management interface"""
    
    user = user_auth.get_current_user()
    user_id = user_auth.get_current_user_id()
    
    if not user or not user_id:
        st.error("Please login to access profile management")
        return
    
    st.header(f"👤 Profile Management - {user['username']}")
    
    # Profile overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Portfolio count
        portfolio_count = len(st.session_state.get('portfolio', []))
        st.metric("Portfolio Positions", portfolio_count)
    
    with col2:
        # Active alerts count
        active_alerts = len([a for a in st.session_state.get('alerts', []) if a.get('status') == 'Active'])
        st.metric("Active Alerts", active_alerts)
    
    with col3:
        # Watchlist size
        watchlist_size = len(st.session_state.get('watchlist', []))
        st.metric("Watchlist Symbols", watchlist_size)
    
    with col4:
        # Member since
        created_date = user['created_at'].strftime('%Y-%m-%d') if user['created_at'] else 'Unknown'
        st.metric("Member Since", created_date)
    
    # Tabs for different profile management sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "⚙️ Preferences", 
        "💾 Data Export", 
        "🗑️ Data Management", 
        "🔧 Account Settings"
    ])
    
    with tab1:
        display_data_overview(user_id)
    
    with tab2:
        display_preferences_management(user_id)
    
    with tab3:
        display_data_export(user_id)
    
    with tab4:
        display_data_management(user_id)
    
    with tab5:
        display_account_settings(user_id, user)

def display_data_overview(user_id: int):
    """Display overview of user's data"""
    
    st.subheader("📊 Your Data Overview")
    
    # Portfolio overview
    st.markdown("### 💼 Portfolio Data")
    portfolio_data = db_manager.get_user_portfolio(user_id)
    
    if portfolio_data:
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df['purchase_date'] = pd.to_datetime(portfolio_df['purchase_date'])
        portfolio_df = portfolio_df.sort_values('created_at', ascending=False)
        
        st.dataframe(
            portfolio_df[['symbol', 'shares', 'purchase_price', 'purchase_value', 'purchase_date']],
            use_container_width=True
        )
    else:
        st.info("No portfolio positions found")
    
    # Alerts overview
    st.markdown("### 🚨 Alerts Data")
    alerts_data = db_manager.get_user_alerts(user_id)
    
    if alerts_data:
        alerts_df = pd.DataFrame(alerts_data)
        alerts_df['created_at'] = pd.to_datetime(alerts_df['created_at'])
        
        # Summary by status
        status_counts = alerts_df['status'].value_counts()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Alerts", status_counts.get('Active', 0))
        with col2:
            st.metric("Triggered Alerts", status_counts.get('Triggered', 0))
        with col3:
            st.metric("Cancelled Alerts", status_counts.get('Cancelled', 0))
        
        # Recent alerts
        st.markdown("**Recent Alerts:**")
        recent_alerts = alerts_df.sort_values('created_at', ascending=False).head(10)
        st.dataframe(
            recent_alerts[['alert_name', 'symbol', 'alert_type', 'status', 'created_at']],
            use_container_width=True
        )
    else:
        st.info("No alerts found")
    
    # Watchlist overview
    st.markdown("### 👁️ Watchlist Data")
    watchlist = db_manager.get_user_watchlist(user_id)
    
    if watchlist:
        st.success(f"You have {len(watchlist)} symbols in your watchlist:")
        watchlist_str = ", ".join(watchlist)
        st.code(watchlist_str)
    else:
        st.info("No watchlist symbols found")
    
    # Preferences overview
    st.markdown("### ⚙️ Preferences Data")
    preferences = db_manager.get_user_preferences(user_id)
    
    if preferences:
        pref_count = sum(len(prefs) for prefs in preferences.values())
        st.success(f"You have {pref_count} preferences saved across {len(preferences)} categories")
        
        for pref_type, prefs in preferences.items():
            with st.expander(f"{pref_type.title()} Preferences ({len(prefs)} items)"):
                for key, value in prefs.items():
                    st.write(f"**{key}:** {value}")
    else:
        st.info("No preferences found")

def display_preferences_management(user_id: int):
    """Display preferences management interface"""
    
    st.subheader("⚙️ Preferences Management")
    
    # Auto-save settings
    st.markdown("### 💾 Auto-Save Settings")
    
    current_auto_save = st.session_state.get('auto_save_enabled', True)
    new_auto_save = st.checkbox(
        "Enable Auto-Save", 
        value=current_auto_save,
        help="Automatically save portfolio, alerts, watchlist, and preferences"
    )
    
    if new_auto_save != current_auto_save:
        st.session_state.auto_save_enabled = new_auto_save
        user_auth.auto_save_preference('system', 'auto_save_enabled', new_auto_save)
        st.success(f"Auto-save {'enabled' if new_auto_save else 'disabled'}")
    
    # Chart preferences
    st.markdown("### 📈 Chart Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_chart_type = user_auth.get_user_preference('charts', 'default_type', 'Candlestick')
        chart_type = st.selectbox(
            "Default Chart Type",
            ["Candlestick", "Line", "OHLC"],
            index=["Candlestick", "Line", "OHLC"].index(default_chart_type)
        )
        
        if chart_type != default_chart_type:
            user_auth.auto_save_preference('charts', 'default_type', chart_type)
            st.success("Chart type preference saved")
    
    with col2:
        default_period = user_auth.get_user_preference('charts', 'default_period', '1y')
        period = st.selectbox(
            "Default Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"].index(default_period)
        )
        
        if period != default_period:
            user_auth.auto_save_preference('charts', 'default_period', period)
            st.success("Time period preference saved")
    
    # Technical indicators preferences
    st.markdown("### 🔍 Technical Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_ma = st.checkbox(
            "Show Moving Averages by default",
            value=user_auth.get_user_preference('indicators', 'default_ma', True)
        )
        user_auth.auto_save_preference('indicators', 'default_ma', show_ma)
    
    with col2:
        show_bb = st.checkbox(
            "Show Bollinger Bands by default",
            value=user_auth.get_user_preference('indicators', 'default_bb', False)
        )
        user_auth.auto_save_preference('indicators', 'default_bb', show_bb)
    
    with col3:
        show_rsi = st.checkbox(
            "Show RSI by default",
            value=user_auth.get_user_preference('indicators', 'default_rsi', False)
        )
        user_auth.auto_save_preference('indicators', 'default_rsi', show_rsi)
    
    # Notification preferences
    st.markdown("### 🔔 Notification Preferences")
    
    default_notifications = user_auth.get_user_preference('notifications', 'default_methods', ['Browser Notification'])
    notification_methods = st.multiselect(
        "Default Notification Methods",
        ["Browser Notification", "Email", "SMS"],
        default=default_notifications
    )
    
    if notification_methods != default_notifications:
        user_auth.auto_save_preference('notifications', 'default_methods', notification_methods)
        st.success("Notification preferences saved")

def display_data_export(user_id: int):
    """Display data export functionality"""
    
    st.subheader("💾 Data Export")
    
    st.markdown("Export your data for backup or analysis purposes:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Portfolio Data", use_container_width=True):
            portfolio_data = db_manager.get_user_portfolio(user_id)
            if portfolio_data:
                df = pd.DataFrame(portfolio_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio CSV",
                    data=csv,
                    file_name=f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No portfolio data to export")
    
    with col2:
        if st.button("🚨 Export Alerts Data", use_container_width=True):
            alerts_data = db_manager.get_user_alerts(user_id)
            if alerts_data:
                # Convert to DataFrame and handle JSON fields
                df = pd.DataFrame(alerts_data)
                df['notification_methods'] = df['notification_methods'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Alerts CSV",
                    data=csv,
                    file_name=f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No alerts data to export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👁️ Export Watchlist", use_container_width=True):
            watchlist = db_manager.get_user_watchlist(user_id)
            if watchlist:
                watchlist_text = '\n'.join(watchlist)
                st.download_button(
                    label="Download Watchlist TXT",
                    data=watchlist_text,
                    file_name=f"watchlist_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No watchlist data to export")
    
    with col2:
        if st.button("⚙️ Export Preferences", use_container_width=True):
            preferences = db_manager.get_user_preferences(user_id)
            if preferences:
                import json
                prefs_json = json.dumps(preferences, indent=2, default=str)
                st.download_button(
                    label="Download Preferences JSON",
                    data=prefs_json,
                    file_name=f"preferences_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No preferences data to export")

def display_data_management(user_id: int):
    """Display data management functionality"""
    
    st.subheader("🗑️ Data Management")
    
    st.warning("⚠️ **Warning:** These actions cannot be undone. Please export your data first if needed.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Clear Individual Data Types")
        
        if st.button("🗑️ Clear Portfolio", use_container_width=True):
            if st.session_state.get('confirm_clear_portfolio'):
                if db_manager.clear_user_portfolio(user_id):
                    st.session_state.portfolio = []
                    st.success("Portfolio cleared successfully")
                    st.session_state.confirm_clear_portfolio = False
                    st.rerun()
            else:
                st.session_state.confirm_clear_portfolio = True
                st.warning("Click again to confirm clearing portfolio")
        
        if st.button("🗑️ Clear Alerts", use_container_width=True):
            if st.session_state.get('confirm_clear_alerts'):
                if db_manager.clear_user_alerts(user_id):
                    st.session_state.alerts = []
                    st.session_state.alert_history = []
                    st.success("Alerts cleared successfully")
                    st.session_state.confirm_clear_alerts = False
                    st.rerun()
            else:
                st.session_state.confirm_clear_alerts = True
                st.warning("Click again to confirm clearing alerts")
    
    with col2:
        st.markdown("### Manual Data Sync")
        
        if st.button("🔄 Reload Data from Database", use_container_width=True):
            user_auth.load_user_data()
            st.success("Data reloaded from database")
            st.rerun()
        
        if st.button("💾 Force Save Current State", use_container_width=True):
            user_auth.save_current_state()
            st.success("Current state saved to database")

def display_account_settings(user_id: int, user: dict):
    """Display account settings"""
    
    st.subheader("🔧 Account Settings")
    
    # Account information
    st.markdown("### 👤 Account Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Username", value=user['username'], disabled=True)
        st.text_input("User ID", value=str(user['id']), disabled=True)
    
    with col2:
        created_at = user['created_at'].strftime('%Y-%m-%d %H:%M:%S') if user['created_at'] else 'Unknown'
        st.text_input("Account Created", value=created_at, disabled=True)
        
        last_login = user['last_login'].strftime('%Y-%m-%d %H:%M:%S') if user.get('last_login') else 'Never'
        st.text_input("Last Login", value=last_login, disabled=True)
    
    # Account actions
    st.markdown("### ⚙️ Account Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Account Data", use_container_width=True):
            # Reload user data
            fresh_user = db_manager.get_user(user['username'])
            if fresh_user:
                st.session_state.user_data = fresh_user
                st.success("Account data refreshed")
                st.rerun()
    
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            user_auth.logout_user()
    
    # Data storage information
    st.markdown("### 📊 Data Storage Statistics")
    
    # Get storage stats
    portfolio_count = len(db_manager.get_user_portfolio(user_id))
    alerts_count = len(db_manager.get_user_alerts(user_id))
    watchlist_count = len(db_manager.get_user_watchlist(user_id))
    preferences = db_manager.get_user_preferences(user_id)
    preferences_count = sum(len(prefs) for prefs in preferences.values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Positions", portfolio_count)
    with col2:
        st.metric("Total Alerts", alerts_count)
    with col3:
        st.metric("Watchlist Symbols", watchlist_count)
    with col4:
        st.metric("Saved Preferences", preferences_count)