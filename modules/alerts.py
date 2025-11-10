import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from modules.user_auth import user_auth
from modules.data_service import get_data_service, format_data_quality_indicator, get_data_quality_color

def display_alerts():
    """Display alerts and notifications system"""
    
    st.subheader("🚨 Alerts & Notifications")
    
    # Initialize session state for alerts
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = []
    
    # Alert management tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔔 Active Alerts", "➕ Create Alert", "📈 Alert Dashboard", "📋 Alert History"])
    
    with tab1:
        display_active_alerts()
    
    with tab2:
        display_create_alert()
    
    with tab3:
        display_alert_dashboard()
    
    with tab4:
        display_alert_history()

def display_active_alerts():
    """Display currently active alerts"""
    
    st.markdown("### 🎯 Active Alerts")
    
    if not st.session_state.alerts:
        st.info("No active alerts. Create your first alert in the 'Create Alert' tab.")
        return
    
    # Check alerts button
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔄 Check All Alerts"):
            with st.spinner("Checking alerts..."):
                triggered_alerts = check_all_alerts()
                if triggered_alerts:
                    st.success(f"Found {len(triggered_alerts)} triggered alerts!")
                    for alert in triggered_alerts:
                        st.warning(f"🚨 ALERT: {alert['message']}")
                else:
                    st.info("No alerts triggered.")
    
    with col2:
        if st.button("🗑️ Clear All Alerts"):
            st.session_state.alerts = []
            st.success("All alerts cleared!")
            st.rerun()
    
    # Display alerts table
    if st.session_state.alerts:
        alerts_df = pd.DataFrame(st.session_state.alerts)
        
        # Format for display
        display_df = alerts_df.copy()
        if 'created_at' in display_df.columns:
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_df, use_container_width=True)

def display_create_alert():
    """Display alert creation interface"""
    
    st.markdown("### ➕ Create New Alert")
    
    # Alert type selection
    alert_type = st.selectbox(
        "Alert Type",
        ["Price Alert", "Volume Alert", "Technical Indicator Alert", "News Alert"]
    )
    
    if alert_type == "Price Alert":
        create_price_alert()
    elif alert_type == "Volume Alert":
        create_volume_alert()
    elif alert_type == "Technical Indicator Alert":
        create_technical_alert()
    elif alert_type == "News Alert":
        create_news_alert()

def create_price_alert():
    """Create price-based alert"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value='AAPL')
        
        condition = st.selectbox(
            "Condition",
            ["Above", "Below", "Between"]
        )
        
        if condition in ["Above", "Below"]:
            target_price = st.number_input(
                f"Target Price",
                value=150.0,
                min_value=0.01,
                step=0.01
            )
        else:
            col1a, col1b = st.columns(2)
            with col1a:
                lower_price = st.number_input("Lower Price", value=140.0, min_value=0.01, step=0.01)
            with col1b:
                upper_price = st.number_input("Upper Price", value=160.0, min_value=0.01, step=0.01)
    
    with col2:
        alert_name = st.text_input("Alert Name", value=f"{symbol} Price Alert")
        
        notification_method = st.multiselect(
            "Notification Method",
            ["Browser Notification", "Email", "SMS"],
            default=["Browser Notification"]
        )
    
    # Get current price for reference using real-time data service
    try:
        data_service = get_data_service()
        quote_data = data_service.get_stock_quote(symbol)
        if quote_data and quote_data.get('price'):
            current_price = quote_data['price']
            data_quality = quote_data.get('data_quality', 'unknown')
            quality_indicator = format_data_quality_indicator(data_quality, quote_data.get('cache_age', 0))
            
            st.info(f"Current price of {symbol}: ${current_price:.2f} | {quality_indicator}")
            
            # Show data source and freshness
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Source: {quote_data.get('source', 'unknown')}")
            with col2:
                if quote_data.get('timestamp'):
                    try:
                        timestamp = datetime.fromisoformat(quote_data['timestamp'].replace('Z', '+00:00'))
                        st.caption(f"Updated: {timestamp.strftime('%H:%M:%S')}")
                    except:
                        st.caption("Updated: Recent")
        else:
            st.warning("Could not fetch current price - using fallback data")
    except Exception as e:
        st.warning(f"Could not fetch current price: {str(e)}")
    
    if st.button("🔔 Create Alert"):
        # Create alert dictionary
        alert = {
            'id': len(st.session_state.alerts),
            'alert_type': 'Price Alert',
            'symbol': symbol.upper(),
            'condition': condition,
            'alert_name': alert_name,
            'notification_method': notification_method,
            'created_at': datetime.now().isoformat(),
            'status': 'Active',
            'triggered_count': 0
        }
        
        # Add price parameters based on condition
        if condition in ["Above", "Below"]:
            alert['target_price'] = target_price
        else:
            alert['lower_price'] = lower_price
            alert['upper_price'] = upper_price
        
        # Auto-save to database
        if user_auth.auto_save_alert(alert):
            st.session_state.alerts.append(alert)
            st.success("Alert created and saved successfully!")
            st.rerun()
        else:
            st.error("Failed to save alert to database")

def create_volume_alert():
    """Create volume-based alert"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value='AAPL')
        volume_multiplier = st.number_input("Volume Multiplier (x average)", value=2.0, min_value=0.1, step=0.1)
    
    with col2:
        alert_name = st.text_input("Alert Name", value=f"{symbol} Volume Alert")
        notification_method = st.multiselect(
            "Notification Method",
            ["Browser Notification", "Email", "SMS"],
            default=["Browser Notification"]
        )
    
    if st.button("🔔 Create Volume Alert"):
        alert = {
            'id': len(st.session_state.alerts),
            'alert_type': 'Volume Alert',
            'symbol': symbol.upper(),
            'volume_multiplier': volume_multiplier,
            'alert_name': alert_name,
            'notification_method': notification_method,
            'created_at': datetime.now().isoformat(),
            'status': 'Active',
            'triggered_count': 0
        }
        
        st.session_state.alerts.append(alert)
        st.success("Volume alert created successfully!")
        st.rerun()

def create_technical_alert():
    """Create technical indicator alert"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value='AAPL')
        indicator = st.selectbox("Technical Indicator", ["RSI", "MACD", "Moving Average Crossover"])
    
    with col2:
        alert_name = st.text_input("Alert Name", value=f"{symbol} {indicator} Alert")
        notification_method = st.multiselect(
            "Notification Method",
            ["Browser Notification", "Email", "SMS"],
            default=["Browser Notification"]
        )
    
    if st.button("🔔 Create Technical Alert"):
        alert = {
            'id': len(st.session_state.alerts),
            'alert_type': 'Technical Indicator Alert',
            'symbol': symbol.upper(),
            'indicator': indicator,
            'alert_name': alert_name,
            'notification_method': notification_method,
            'created_at': datetime.now().isoformat(),
            'status': 'Active',
            'triggered_count': 0
        }
        
        st.session_state.alerts.append(alert)
        st.success("Technical indicator alert created successfully!")
        st.rerun()

def create_news_alert():
    """Create news-based alert"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol (optional)", value='', help="Leave empty for general market news")
        keywords = st.text_input("Keywords (comma-separated)", value='earnings, acquisition, merger')
    
    with col2:
        alert_name = st.text_input("Alert Name", value=f"News Alert - {symbol or 'Market'}")
        notification_method = st.multiselect(
            "Notification Method",
            ["Browser Notification", "Email", "SMS"],
            default=["Browser Notification"]
        )
    
    if st.button("🔔 Create News Alert"):
        alert = {
            'id': len(st.session_state.alerts),
            'alert_type': 'News Alert',
            'symbol': symbol.upper() if symbol else '',
            'keywords': keywords,
            'alert_name': alert_name,
            'notification_method': notification_method,
            'created_at': datetime.now().isoformat(),
            'status': 'Active',
            'triggered_count': 0
        }
        
        st.session_state.alerts.append(alert)
        st.success("News alert created successfully!")
        st.rerun()

def display_alert_dashboard():
    """Display alert dashboard with analytics"""
    
    st.markdown("### 📊 Alert Dashboard")
    
    if not st.session_state.alerts:
        st.info("No alerts to display in dashboard.")
        return
    
    # Alert statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_alerts = len(st.session_state.alerts)
    active_alerts = len([a for a in st.session_state.alerts if a.get('status') == 'Active'])
    total_triggered = sum(a.get('triggered_count', 0) for a in st.session_state.alerts)
    alert_types = len(set(a['alert_type'] for a in st.session_state.alerts))
    
    with col1:
        st.metric("Total Alerts", total_alerts)
    
    with col2:
        st.metric("Active Alerts", active_alerts)
    
    with col3:
        st.metric("Total Triggered", total_triggered)
    
    with col4:
        st.metric("Alert Types", alert_types)
    
    # Alert types distribution
    alert_type_counts = pd.Series([a['alert_type'] for a in st.session_state.alerts]).value_counts()
    
    fig_types = px.pie(
        values=alert_type_counts.values,
        names=alert_type_counts.index,
        title="Alert Types Distribution"
    )
    
    fig_types.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_types, use_container_width=True)
    
    # Most watched symbols
    symbols = [a.get('symbol', 'N/A') for a in st.session_state.alerts if a.get('symbol')]
    if symbols:
        symbol_counts = pd.Series(symbols).value_counts().head(10)
        
        st.markdown("**Most Watched Symbols:**")
        
        fig_symbols = px.bar(
            x=symbol_counts.values,
            y=symbol_counts.index,
            orientation='h',
            title="Top 10 Most Watched Symbols"
        )
        
        fig_symbols.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_symbols, use_container_width=True)

def display_alert_history():
    """Display alert history and triggered alerts"""
    
    st.markdown("### 📋 Alert History")
    
    if not st.session_state.alert_history:
        st.info("No alert history available.")
        return
    
    # Display history
    history_df = pd.DataFrame(st.session_state.alert_history)
    history_df['triggered_at'] = pd.to_datetime(history_df['triggered_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(history_df, use_container_width=True)
    
    # Clear history button
    if st.button("🗑️ Clear Alert History"):
        st.session_state.alert_history = []
        st.success("Alert history cleared!")
        st.rerun()

def check_all_alerts():
    """Check all active alerts and return triggered ones"""
    
    triggered_alerts = []
    
    for i, alert in enumerate(st.session_state.alerts):
        if alert.get('status') != 'Active':
            continue
        
        triggered = False
        message = ""
        
        try:
            if alert['alert_type'] == 'Price Alert':
                triggered, message = check_price_alert(alert)
            elif alert['alert_type'] == 'Volume Alert':
                triggered, message = check_volume_alert(alert)
            elif alert['alert_type'] == 'Technical Indicator Alert':
                triggered, message = check_technical_alert(alert)
            elif alert['alert_type'] == 'News Alert':
                triggered, message = check_news_alert(alert)
            
            if triggered:
                # Update alert
                st.session_state.alerts[i]['triggered_count'] = alert.get('triggered_count', 0) + 1
                st.session_state.alerts[i]['last_triggered'] = datetime.now().isoformat()
                
                # Add to history
                history_entry = alert.copy()
                history_entry['triggered_at'] = datetime.now().isoformat()
                history_entry['message'] = message
                st.session_state.alert_history.append(history_entry)
                
                # Add to triggered list
                triggered_alerts.append({
                    'alert': alert,
                    'message': message
                })
        
        except Exception as e:
            continue  # Skip alerts that fail to check
    
    return triggered_alerts

def check_price_alert(alert):
    """Check if price alert should trigger using real-time data"""
    
    try:
        symbol = alert['symbol']
        data_service = get_data_service()
        quote_data = data_service.get_stock_quote(symbol, force_refresh=True)  # Force refresh for alerts
        
        if not quote_data or not quote_data.get('price'):
            return False, "Could not fetch current price"
        
        current_price = quote_data['price']
        data_quality = quote_data.get('data_quality', 'unknown')
        condition = alert['condition']
        
        # Add data quality to alert message for transparency
        quality_suffix = f" [{format_data_quality_indicator(data_quality, quote_data.get('cache_age', 0))}]"
        
        if condition == "Above":
            target_price = alert['target_price']
            if current_price > target_price:
                return True, f"{symbol} is above ${target_price:.2f} (Current: ${current_price:.2f}){quality_suffix}"
        
        elif condition == "Below":
            target_price = alert['target_price']
            if current_price < target_price:
                return True, f"{symbol} is below ${target_price:.2f} (Current: ${current_price:.2f}){quality_suffix}"
        
        elif condition == "Between":
            lower_price = alert['lower_price']
            upper_price = alert['upper_price']
            if lower_price <= current_price <= upper_price:
                return True, f"{symbol} is between ${lower_price:.2f} and ${upper_price:.2f} (Current: ${current_price:.2f}){quality_suffix}"
        
        return False, ""
        
    except Exception as e:
        return False, f"Error checking price alert: {str(e)}"

def check_volume_alert(alert):
    """Check if volume alert should trigger using real-time data"""
    
    try:
        symbol = alert['symbol']
        data_service = get_data_service()
        
        # Get current quote for volume
        quote_data = data_service.get_stock_quote(symbol, force_refresh=True)
        if not quote_data or not quote_data.get('volume'):
            return False, "Could not fetch current volume data"
        
        current_volume = quote_data['volume']
        
        # Get historical data to calculate average volume
        hist_data = data_service.get_historical_data(symbol, period="5d")
        if hist_data is None or hist_data.empty:
            return False, "Could not fetch historical volume data for comparison"
        
        avg_volume = hist_data['Volume'].mean()
        multiplier = alert.get('volume_multiplier', 2.0)
        
        if current_volume > avg_volume * multiplier:
            quality_indicator = format_data_quality_indicator(
                quote_data.get('data_quality', 'unknown'), 
                quote_data.get('cache_age', 0)
            )
            return True, f"{symbol} volume is {multiplier}x above average (Current: {current_volume:,.0f}, Avg: {avg_volume:,.0f}) [{quality_indicator}]"
        
        return False, ""
        
    except Exception as e:
        return False, f"Error checking volume alert: {str(e)}"

def check_technical_alert(alert):
    """Check if technical indicator alert should trigger"""
    
    # Simplified check for demo purposes
    if np.random.random() > 0.98:  # 2% chance for demo
        symbol = alert.get('symbol', 'UNKNOWN')
        indicator = alert.get('indicator', 'UNKNOWN')
        return True, f"{symbol} {indicator} indicator triggered"
    
    return False, ""

def check_news_alert(alert):
    """Check if news alert should trigger"""
    
    # Simplified check for demo purposes
    if np.random.random() > 0.99:  # 1% chance for demo
        symbol = alert.get('symbol', 'Market')
        keywords = alert.get('keywords', 'news')
        return True, f"News alert triggered for {symbol}: {keywords.split(',')[0].strip()}"
    
    return False, ""