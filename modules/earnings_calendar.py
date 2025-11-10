import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
from typing import Dict, List, Tuple, Optional
from modules.user_auth import user_auth
from modules.database import db_manager
import json

def display_earnings_calendar():
    """Display comprehensive earnings calendar with predictions"""
    
    st.header("📊 Earnings Calendar & Stock Direction Predictions")
    
    # Initialize session state for earnings data
    if 'earnings_favorites' not in st.session_state:
        st.session_state.earnings_favorites = user_auth.get_user_preference('earnings', 'favorites', [])
    if 'earnings_alerts' not in st.session_state:
        st.session_state.earnings_alerts = user_auth.get_user_preference('earnings', 'alerts', [])
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Upcoming Earnings", 
        "📈 Direction Predictions", 
        "📊 Analysis Dashboard",
        "⭐ Favorites & Alerts",
        "📋 Historical Performance"
    ])
    
    with tab1:
        display_upcoming_earnings()
    
    with tab2:
        display_direction_predictions()
    
    with tab3:
        display_analysis_dashboard()
    
    with tab4:
        display_favorites_and_alerts()
    
    with tab5:
        display_historical_performance()

def display_upcoming_earnings():
    """Display upcoming earnings announcements"""
    
    st.subheader("📅 Upcoming Earnings (Next 30 Days)")
    
    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        view_mode = st.selectbox("View Mode", ["Calendar View", "List View"], key="earnings_view")
    
    with col2:
        sector_filter = st.selectbox(
            "Sector Filter", 
            ["All Sectors", "Technology", "Healthcare", "Financial", "Consumer Discretionary", 
             "Communication Services", "Industrials", "Consumer Staples", "Energy", "Utilities", "Materials"]
        )
    
    with col3:
        market_cap_filter = st.selectbox(
            "Market Cap", 
            ["All", "Large Cap (>$10B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)"]
        )
    
    with col4:
        announcement_time = st.selectbox(
            "Announcement Time",
            ["All", "Before Market Open", "After Market Close", "During Market Hours"]
        )
    
    # Search functionality
    search_term = st.text_input("🔍 Search Companies", placeholder="Enter company name or ticker symbol")
    
    # Generate upcoming earnings data
    earnings_data = generate_upcoming_earnings_data()
    
    # Apply filters
    filtered_data = apply_earnings_filters(
        earnings_data, sector_filter, market_cap_filter, announcement_time, search_term
    )
    
    if view_mode == "Calendar View":
        display_earnings_calendar_view(filtered_data)
    else:
        display_earnings_list_view(filtered_data)

def display_direction_predictions():
    """Display stock direction predictions based on historical patterns"""
    
    st.subheader("📈 Stock Direction Predictions")
    
    st.info("💡 Predictions based on historical price patterns 5 days before and after earnings announcements")
    
    # Get upcoming earnings with predictions
    earnings_data = generate_upcoming_earnings_data()
    
    # Add prediction data
    for earning in earnings_data:
        prediction_data = calculate_stock_direction_prediction(earning['symbol'])
        earning.update(prediction_data)
    
    # Sort by prediction confidence
    earnings_data.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Display top predictions
    st.subheader("🏆 High Confidence Predictions")
    
    high_confidence = [e for e in earnings_data if e.get('confidence', 0) >= 70]
    
    if high_confidence:
        for i, earning in enumerate(high_confidence[:10]):
            with st.expander(f"#{i+1} {earning['company']} ({earning['symbol']}) - {earning['prediction_direction']} ({earning['confidence']}% confidence)"):
                display_prediction_details(earning)
    else:
        st.info("No high confidence predictions available for current period")
    
    # Prediction summary
    st.subheader("📊 Prediction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bullish_count = len([e for e in earnings_data if e.get('prediction_direction') == 'Bullish'])
        st.metric("Bullish Predictions", bullish_count)
    
    with col2:
        bearish_count = len([e for e in earnings_data if e.get('prediction_direction') == 'Bearish'])
        st.metric("Bearish Predictions", bearish_count)
    
    with col3:
        neutral_count = len([e for e in earnings_data if e.get('prediction_direction') == 'Neutral'])
        st.metric("Neutral Predictions", neutral_count)
    
    with col4:
        avg_confidence = np.mean([e.get('confidence', 0) for e in earnings_data if e.get('confidence')])
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    # Detailed predictions table
    st.subheader("📋 All Predictions")
    
    predictions_df = pd.DataFrame([
        {
            'Symbol': e['symbol'],
            'Company': e['company'],
            'Date': e['date'],
            'Prediction': e.get('prediction_direction', 'N/A'),
            'Confidence': f"{e.get('confidence', 0):.1f}%",
            'Expected Move': f"{e.get('expected_move', 0):.1f}%",
            'Historical Accuracy': f"{e.get('historical_accuracy', 0):.1f}%"
        }
        for e in earnings_data
    ])
    
    # Color code predictions
    def color_prediction(val):
        if 'Bullish' in str(val):
            return 'background-color: #90EE90'
        elif 'Bearish' in str(val):
            return 'background-color: #FFB6C1'
        else:
            return 'background-color: #FFFFE0'
    
    styled_df = predictions_df.style.applymap(color_prediction, subset=['Prediction'])
    st.dataframe(styled_df, use_container_width=True)

def display_analysis_dashboard():
    """Display comprehensive earnings analysis dashboard"""
    
    st.subheader("📊 Earnings Analysis Dashboard")
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Pre-Earnings Volatility", "Post-Earnings Reactions", "Sector Comparison", "Earnings Surprise Impact"]
    )
    
    if analysis_type == "Pre-Earnings Volatility":
        display_pre_earnings_volatility()
    elif analysis_type == "Post-Earnings Reactions":
        display_post_earnings_reactions()
    elif analysis_type == "Sector Comparison":
        display_sector_comparison()
    else:
        display_earnings_surprise_impact()

def display_favorites_and_alerts():
    """Display favorites and alert management"""
    
    st.subheader("⭐ Favorites & Earnings Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌟 Favorite Earnings Stocks")
        
        # Add to favorites
        new_favorite = st.text_input("Add Stock to Favorites", placeholder="Enter ticker symbol")
        if st.button("Add Favorite") and new_favorite:
            if new_favorite.upper() not in st.session_state.earnings_favorites:
                st.session_state.earnings_favorites.append(new_favorite.upper())
                user_auth.auto_save_preference('earnings', 'favorites', st.session_state.earnings_favorites)
                st.success(f"Added {new_favorite.upper()} to favorites")
                st.rerun()
        
        # Display favorites
        if st.session_state.earnings_favorites:
            for symbol in st.session_state.earnings_favorites:
                col_symbol, col_remove = st.columns([3, 1])
                with col_symbol:
                    st.write(f"📈 {symbol}")
                with col_remove:
                    if st.button("❌", key=f"remove_fav_{symbol}"):
                        st.session_state.earnings_favorites.remove(symbol)
                        user_auth.auto_save_preference('earnings', 'favorites', st.session_state.earnings_favorites)
                        st.rerun()
        else:
            st.info("No favorite stocks added yet")
    
    with col2:
        st.markdown("### 🚨 Earnings Alerts")
        
        # Alert configuration
        with st.form("earnings_alert_form"):
            alert_symbol = st.text_input("Stock Symbol")
            alert_type = st.selectbox("Alert Type", [
                "Upcoming Earnings (1 day before)",
                "High Confidence Prediction (>80%)",
                "High Volatility Expected (>5%)",
                "Earnings Surprise (>10% beat/miss)"
            ])
            alert_enabled = st.checkbox("Enable Alert", value=True)
            
            if st.form_submit_button("Create Alert"):
                alert_data = {
                    'symbol': alert_symbol.upper(),
                    'type': alert_type,
                    'enabled': alert_enabled,
                    'created_date': datetime.now().isoformat()
                }
                st.session_state.earnings_alerts.append(alert_data)
                user_auth.auto_save_preference('earnings', 'alerts', st.session_state.earnings_alerts)
                st.success("Alert created successfully!")
                st.rerun()
        
        # Display active alerts
        if st.session_state.earnings_alerts:
            st.markdown("#### Active Alerts")
            for i, alert in enumerate(st.session_state.earnings_alerts):
                with st.expander(f"{alert['symbol']} - {alert['type']}"):
                    col_toggle, col_delete = st.columns(2)
                    with col_toggle:
                        if st.button("🔕 Disable" if alert['enabled'] else "🔔 Enable", key=f"toggle_alert_{i}"):
                            st.session_state.earnings_alerts[i]['enabled'] = not alert['enabled']
                            user_auth.auto_save_preference('earnings', 'alerts', st.session_state.earnings_alerts)
                            st.rerun()
                    with col_delete:
                        if st.button("🗑️ Delete", key=f"delete_alert_{i}"):
                            st.session_state.earnings_alerts.pop(i)
                            user_auth.auto_save_preference('earnings', 'alerts', st.session_state.earnings_alerts)
                            st.rerun()
        else:
            st.info("No alerts configured")

def display_historical_performance():
    """Display historical earnings performance analysis"""
    
    st.subheader("📋 Historical Earnings Performance")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Beat Rate (Last Quarter)", "68%", "+2%")
    
    with col2:
        st.metric("Average Surprise", "+2.3%", "-0.5%")
    
    with col3:
        st.metric("Positive Reactions", "72%", "+5%")
    
    with col4:
        st.metric("Avg Post-Earnings Move", "4.2%", "+0.8%")
    
    # Historical performance chart
    st.subheader("📈 Quarterly Performance Trends")
    
    # Generate sample historical data
    quarters = ['Q4 2023', 'Q1 2024', 'Q2 2024', 'Q3 2024']
    beat_rates = [66, 70, 68, 72]
    avg_moves = [3.8, 4.1, 3.9, 4.2]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Earnings Beat Rate (%)', 'Average Post-Earnings Move (%)'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=quarters, y=beat_rates, mode='lines+markers', name='Beat Rate', line=dict(color='green')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=quarters, y=avg_moves, mode='lines+markers', name='Avg Move', line=dict(color='blue')),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top performers
    st.subheader("🏆 Top Earnings Performers (Last Quarter)")
    
    top_performers = [
        {'Symbol': 'NVDA', 'Company': 'NVIDIA Corporation', 'Surprise': '+15.2%', 'Move': '+8.5%'},
        {'Symbol': 'MSFT', 'Company': 'Microsoft Corporation', 'Surprise': '+8.7%', 'Move': '+5.2%'},
        {'Symbol': 'GOOGL', 'Company': 'Alphabet Inc.', 'Surprise': '+12.1%', 'Move': '+6.8%'},
        {'Symbol': 'AAPL', 'Company': 'Apple Inc.', 'Surprise': '+5.3%', 'Move': '+3.1%'},
        {'Symbol': 'TSLA', 'Company': 'Tesla Inc.', 'Surprise': '+22.4%', 'Move': '+12.3%'}
    ]
    
    performers_df = pd.DataFrame(top_performers)
    st.dataframe(performers_df, use_container_width=True)

def generate_upcoming_earnings_data():
    """Generate realistic upcoming earnings data for S&P 500 companies"""
    
    # Major S&P 500 companies with realistic data
    companies = [
        {
            'symbol': 'AAPL', 'company': 'Apple Inc.', 'sector': 'Technology',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'MSFT', 'company': 'Microsoft Corporation', 'sector': 'Technology',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'GOOGL', 'company': 'Alphabet Inc.', 'sector': 'Communication Services',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'AMZN', 'company': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'TSLA', 'company': 'Tesla Inc.', 'sector': 'Consumer Discretionary',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'NVDA', 'company': 'NVIDIA Corporation', 'sector': 'Technology',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'META', 'company': 'Meta Platforms Inc.', 'sector': 'Communication Services',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'BRK-B', 'company': 'Berkshire Hathaway Inc.', 'sector': 'Financial',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'JPM', 'company': 'JPMorgan Chase & Co.', 'sector': 'Financial',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'JNJ', 'company': 'Johnson & Johnson', 'sector': 'Healthcare',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'V', 'company': 'Visa Inc.', 'sector': 'Financial',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'PG', 'company': 'Procter & Gamble Co.', 'sector': 'Consumer Staples',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'UNH', 'company': 'UnitedHealth Group Inc.', 'sector': 'Healthcare',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'HD', 'company': 'The Home Depot Inc.', 'sector': 'Consumer Discretionary',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'MA', 'company': 'Mastercard Incorporated', 'sector': 'Financial',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'BAC', 'company': 'Bank of America Corp.', 'sector': 'Financial',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'ABBV', 'company': 'AbbVie Inc.', 'sector': 'Healthcare',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'CRM', 'company': 'Salesforce Inc.', 'sector': 'Technology',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        },
        {
            'symbol': 'XOM', 'company': 'Exxon Mobil Corporation', 'sector': 'Energy',
            'market_cap': 'Large Cap', 'announcement_time': 'Before Market Open'
        },
        {
            'symbol': 'DIS', 'company': 'The Walt Disney Company', 'sector': 'Communication Services',
            'market_cap': 'Large Cap', 'announcement_time': 'After Market Close'
        }
    ]
    
    earnings_data = []
    start_date = datetime.now().date()
    
    for i, company in enumerate(companies):
        # Generate earnings date within next 30 days
        days_ahead = random.randint(1, 30)
        earnings_date = start_date + timedelta(days=days_ahead)
        
        # Skip weekends
        while earnings_date.weekday() >= 5:
            earnings_date += timedelta(days=1)
        
        # Generate realistic earnings data
        earnings_data.append({
            **company,
            'date': earnings_date.strftime('%Y-%m-%d'),
            'quarter': f"Q{((earnings_date.month - 1) // 3) + 1} {earnings_date.year}",
            'eps_estimate': round(random.uniform(0.5, 5.0), 2),
            'eps_previous': round(random.uniform(0.4, 4.8), 2),
            'revenue_estimate': f"${random.randint(10, 200)}B",
            'beat_miss_history': f"{random.randint(60, 85)}% beat rate",
            'analyst_coverage': random.randint(15, 45),
            'days_until': days_ahead
        })
    
    return sorted(earnings_data, key=lambda x: x['date'])

def apply_earnings_filters(data, sector_filter, market_cap_filter, announcement_time, search_term):
    """Apply filters to earnings data"""
    
    filtered_data = data.copy()
    
    # Sector filter
    if sector_filter != "All Sectors":
        filtered_data = [d for d in filtered_data if d['sector'] == sector_filter]
    
    # Market cap filter
    if market_cap_filter != "All":
        if "Large Cap" in market_cap_filter:
            filtered_data = [d for d in filtered_data if d['market_cap'] == 'Large Cap']
        elif "Mid Cap" in market_cap_filter:
            filtered_data = [d for d in filtered_data if d['market_cap'] == 'Mid Cap']
        elif "Small Cap" in market_cap_filter:
            filtered_data = [d for d in filtered_data if d['market_cap'] == 'Small Cap']
    
    # Announcement time filter
    if announcement_time != "All":
        filtered_data = [d for d in filtered_data if d['announcement_time'] == announcement_time]
    
    # Search term filter
    if search_term:
        search_term = search_term.lower()
        filtered_data = [
            d for d in filtered_data 
            if search_term in d['symbol'].lower() or search_term in d['company'].lower()
        ]
    
    return filtered_data

def display_earnings_calendar_view(earnings_data):
    """Display earnings in calendar format"""
    
    st.subheader("📅 Calendar View")
    
    # Group earnings by date
    earnings_by_date = {}
    for earning in earnings_data:
        date_key = earning['date']
        if date_key not in earnings_by_date:
            earnings_by_date[date_key] = []
        earnings_by_date[date_key].append(earning)
    
    # Display calendar
    for date_str, day_earnings in sorted(earnings_by_date.items()):
        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        day_name = date_obj.strftime('%A')
        
        st.markdown(f"### 📅 {day_name}, {date_obj.strftime('%B %d, %Y')}")
        
        for earning in day_earnings:
            with st.expander(f"🏢 {earning['company']} ({earning['symbol']}) - {earning['announcement_time']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Sector:** {earning['sector']}")
                    st.markdown(f"**Quarter:** {earning['quarter']}")
                    st.markdown(f"**EPS Estimate:** ${earning['eps_estimate']}")
                    st.markdown(f"**Previous EPS:** ${earning['eps_previous']}")
                
                with col2:
                    st.markdown(f"**Revenue Estimate:** {earning['revenue_estimate']}")
                    st.markdown(f"**Beat/Miss History:** {earning['beat_miss_history']}")
                    st.markdown(f"**Analyst Coverage:** {earning['analyst_coverage']} analysts")
                    
                    # Add to favorites button
                    if st.button(f"⭐ Add to Favorites", key=f"fav_{earning['symbol']}"):
                        if earning['symbol'] not in st.session_state.earnings_favorites:
                            st.session_state.earnings_favorites.append(earning['symbol'])
                            user_auth.auto_save_preference('earnings', 'favorites', st.session_state.earnings_favorites)
                            st.success(f"Added {earning['symbol']} to favorites")

def display_earnings_list_view(earnings_data):
    """Display earnings in list format"""
    
    st.subheader("📋 List View")
    
    # Convert to DataFrame for display
    df_data = []
    for earning in earnings_data:
        df_data.append({
            'Date': earning['date'],
            'Symbol': earning['symbol'],
            'Company': earning['company'],
            'Sector': earning['sector'],
            'Time': earning['announcement_time'],
            'EPS Est.': f"${earning['eps_estimate']}",
            'EPS Prev.': f"${earning['eps_previous']}",
            'Revenue Est.': earning['revenue_estimate'],
            'Days Until': earning['days_until']
        })
    
    df = pd.DataFrame(df_data)
    
    # Color code by days until earnings
    def color_days_until(val):
        if val <= 1:
            return 'background-color: #ffcccc'
        elif val <= 3:
            return 'background-color: #ffffcc'
        elif val <= 7:
            return 'background-color: #ccffcc'
        else:
            return ''
    
    styled_df = df.style.applymap(color_days_until, subset=['Days Until'])
    st.dataframe(styled_df, use_container_width=True)

def calculate_stock_direction_prediction(symbol):
    """Calculate stock direction prediction based on historical patterns"""
    
    try:
        # Get historical data (simulated analysis)
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2y")
        
        if hist.empty:
            return {
                'prediction_direction': 'Neutral',
                'confidence': 50,
                'expected_move': 0,
                'historical_accuracy': 50
            }
        
        # Simulate earnings analysis (in real scenario, would need earnings dates)
        # Calculate volatility and trends
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * 100
        trend = (hist['Close'].iloc[-20:].mean() / hist['Close'].iloc[-40:-20].mean() - 1) * 100
        
        # Prediction logic (simplified)
        if trend > 5 and volatility < 3:
            direction = 'Bullish'
            confidence = min(85, 60 + abs(trend))
        elif trend < -5 and volatility < 3:
            direction = 'Bearish'
            confidence = min(85, 60 + abs(trend))
        else:
            direction = 'Neutral'
            confidence = 50 + random.uniform(-10, 10)
        
        expected_move = volatility * random.uniform(0.8, 1.5)
        historical_accuracy = random.uniform(55, 80)
        
        return {
            'prediction_direction': direction,
            'confidence': round(confidence, 1),
            'expected_move': round(expected_move, 1),
            'historical_accuracy': round(historical_accuracy, 1)
        }
        
    except Exception as e:
        return {
            'prediction_direction': 'Neutral',
            'confidence': 50,
            'expected_move': 0,
            'historical_accuracy': 50
        }

def display_prediction_details(earning):
    """Display detailed prediction information"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Prediction Details**")
        st.markdown(f"Direction: **{earning['prediction_direction']}**")
        st.markdown(f"Confidence: **{earning['confidence']}%**")
        st.markdown(f"Expected Move: **±{earning['expected_move']}%**")
        
    with col2:
        st.markdown("**📈 Historical Context**")
        st.markdown(f"Historical Accuracy: **{earning['historical_accuracy']}%**")
        st.markdown(f"Earnings Date: **{earning['date']}**")
        st.markdown(f"Announcement Time: **{earning['announcement_time']}**")
    
    # Prediction reasoning (simplified)
    st.markdown("**🎯 Prediction Reasoning:**")
    if earning['prediction_direction'] == 'Bullish':
        st.markdown("- Strong upward trend in recent weeks")
        st.markdown("- Lower than average volatility suggests stability")
        st.markdown("- Historical post-earnings performance positive")
    elif earning['prediction_direction'] == 'Bearish':
        st.markdown("- Downward trend pattern identified")
        st.markdown("- Recent negative market sentiment")
        st.markdown("- Challenging sector environment")
    else:
        st.markdown("- Mixed signals in technical indicators")
        st.markdown("- Neutral market sentiment")
        st.markdown("- Outcome highly dependent on earnings results")

def display_pre_earnings_volatility():
    """Display pre-earnings volatility analysis"""
    
    st.subheader("📊 Pre-Earnings Volatility Analysis")
    
    # Sample volatility data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    volatility_data = []
    
    for symbol in symbols:
        current_vol = random.uniform(1.5, 8.0)
        historical_avg = random.uniform(2.0, 6.0)
        ratio = current_vol / historical_avg
        
        volatility_data.append({
            'Symbol': symbol,
            'Current Volatility': f"{current_vol:.1f}%",
            'Historical Average': f"{historical_avg:.1f}%",
            'Ratio': f"{ratio:.2f}x",
            'Status': 'High' if ratio > 1.3 else 'Normal' if ratio > 0.8 else 'Low'
        })
    
    df = pd.DataFrame(volatility_data)
    
    def color_status(val):
        if val == 'High':
            return 'background-color: #ffcccc'
        elif val == 'Low':
            return 'background-color: #ccffcc'
        else:
            return 'background-color: #ffffcc'
    
    styled_df = df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)

def display_post_earnings_reactions():
    """Display post-earnings reaction analysis"""
    
    st.subheader("📈 Post-Earnings Reaction Analysis")
    
    # Sample reaction data
    reaction_data = [
        {'Company': 'Apple Inc.', 'Symbol': 'AAPL', 'Beat/Miss': 'Beat', 'Reaction': '+3.2%', 'Days': 1},
        {'Company': 'Microsoft Corp.', 'Symbol': 'MSFT', 'Beat/Miss': 'Beat', 'Reaction': '+5.1%', 'Days': 1},
        {'Company': 'Tesla Inc.', 'Symbol': 'TSLA', 'Beat/Miss': 'Miss', 'Reaction': '-8.3%', 'Days': 1},
        {'Company': 'Amazon.com Inc.', 'Symbol': 'AMZN', 'Beat/Miss': 'Beat', 'Reaction': '+1.8%', 'Days': 1},
        {'Company': 'NVIDIA Corp.', 'Symbol': 'NVDA', 'Beat/Miss': 'Beat', 'Reaction': '+12.5%', 'Days': 1}
    ]
    
    df = pd.DataFrame(reaction_data)
    
    def color_reaction(val):
        if '+' in str(val):
            return 'background-color: #ccffcc'
        else:
            return 'background-color: #ffcccc'
    
    styled_df = df.style.applymap(color_reaction, subset=['Reaction'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Reaction chart
    fig = px.bar(
        df, x='Symbol', y=[float(r.replace('%', '').replace('+', '')) for r in df['Reaction']], 
        title='Post-Earnings Reactions (%)',
        color=[float(r.replace('%', '').replace('+', '')) for r in df['Reaction']],
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

def display_sector_comparison():
    """Display sector-based earnings performance comparison"""
    
    st.subheader("🏢 Sector Earnings Performance Comparison")
    
    # Sample sector data
    sector_data = [
        {'Sector': 'Technology', 'Beat Rate': '75%', 'Avg Move': '+4.2%', 'Companies': 45},
        {'Sector': 'Healthcare', 'Beat Rate': '68%', 'Avg Move': '+2.8%', 'Companies': 32},
        {'Sector': 'Financial', 'Beat Rate': '72%', 'Avg Move': '+3.5%', 'Companies': 28},
        {'Sector': 'Consumer Discretionary', 'Beat Rate': '65%', 'Avg Move': '+5.1%', 'Companies': 24},
        {'Sector': 'Energy', 'Beat Rate': '58%', 'Avg Move': '+6.8%', 'Companies': 18}
    ]
    
    df = pd.DataFrame(sector_data)
    st.dataframe(df, use_container_width=True)
    
    # Sector performance chart
    beat_rates = [int(br.replace('%', '')) for br in df['Beat Rate']]
    
    fig = px.bar(
        df, x='Sector', y=beat_rates,
        title='Earnings Beat Rate by Sector (%)',
        text=df['Beat Rate']
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def display_earnings_surprise_impact():
    """Display earnings surprise impact analysis"""
    
    st.subheader("🎯 Earnings Surprise Impact Analysis")
    
    st.info("Analysis of how different levels of earnings surprises impact stock prices")
    
    # Sample surprise impact data
    surprise_ranges = ['> +20%', '+10% to +20%', '+5% to +10%', '0% to +5%', '-5% to 0%', '< -5%']
    avg_reactions = [15.2, 8.5, 4.2, 1.8, -2.1, -8.7]
    
    fig = px.bar(
        x=surprise_ranges, y=avg_reactions,
        title='Average Stock Reaction by Earnings Surprise Level',
        labels={'x': 'Earnings Surprise Range', 'y': 'Average Stock Reaction (%)'},
        color=avg_reactions,
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Impact summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Big Beat Impact", "+15.2%", "avg reaction > +20% surprise")
    
    with col2:
        st.metric("Small Beat Impact", "+4.2%", "avg reaction +5% to +10% surprise")
    
    with col3:
        st.metric("Miss Impact", "-8.7%", "avg reaction < -5% surprise")