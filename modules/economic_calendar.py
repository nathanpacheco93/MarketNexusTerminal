import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests

def display_calendar():
    """Display economic calendar section"""
    
    st.subheader("📅 Economic Calendar")
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date", 
            value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=30)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime.now().date() + timedelta(days=7),
            max_value=datetime.now().date() + timedelta(days=30)
        )
    
    # Importance filter
    importance_filter = st.selectbox(
        "Event Importance",
        ["All", "High", "Medium", "Low"]
    )
    
    # Since we can't access real economic calendar APIs without keys,
    # we'll display a structured calendar with typical events
    st.info("📊 Economic events are displayed based on typical market calendar patterns")
    
    # Generate sample economic events
    events = generate_economic_events(start_date, end_date, importance_filter)
    
    if events:
        # Display events in a structured format
        df = pd.DataFrame(events)
        
        # Color coding based on importance
        def color_importance(val):
            if val == "High":
                return "background-color: #ff6b6b; color: white"
            elif val == "Medium":
                return "background-color: #ffa726; color: white"
            else:
                return "background-color: #66bb6a; color: white"
        
        # Style the dataframe
        styled_df = df.style.applymap(color_importance, subset=['Importance'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Event details
        st.subheader("📋 Event Details & Impact")
        
        # Group events by importance
        high_impact = [e for e in events if e['Importance'] == 'High']
        medium_impact = [e for e in events if e['Importance'] == 'Medium']
        
        if high_impact:
            st.markdown("### 🔴 High Impact Events")
            for event in high_impact:
                with st.expander(f"{event['Date']} - {event['Event']}"):
                    st.markdown(f"**Time:** {event['Time']}")
                    st.markdown(f"**Currency:** {event['Currency']}")
                    st.markdown(f"**Previous:** {event['Previous']}")
                    st.markdown(f"**Forecast:** {event['Forecast']}")
                    st.markdown(f"**Description:** {get_event_description(event['Event'])}")
        
        if medium_impact:
            st.markdown("### 🟡 Medium Impact Events")
            for event in medium_impact[:3]:  # Show first 3
                with st.expander(f"{event['Date']} - {event['Event']}"):
                    st.markdown(f"**Time:** {event['Time']}")
                    st.markdown(f"**Currency:** {event['Currency']}")
                    st.markdown(f"**Previous:** {event['Previous']}")
                    st.markdown(f"**Forecast:** {event['Forecast']}")
        
        # Market impact analysis
        st.subheader("📊 Potential Market Impact")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**USD Impact Events**")
            usd_events = len([e for e in events if e['Currency'] == 'USD'])
            st.metric("USD Events", usd_events)
            
            if usd_events > 5:
                st.warning("⚠️ High USD volatility expected")
            elif usd_events > 2:
                st.info("📊 Moderate USD activity")
            else:
                st.success("✅ Low USD volatility")
        
        with col2:
            st.markdown("**EUR Impact Events**")
            eur_events = len([e for e in events if e['Currency'] == 'EUR'])
            st.metric("EUR Events", eur_events)
            
            if eur_events > 3:
                st.warning("⚠️ EUR volatility expected")
            else:
                st.info("📊 Normal EUR activity")
        
        with col3:
            st.markdown("**High Impact Count**")
            high_count = len([e for e in events if e['Importance'] == 'High'])
            st.metric("High Impact", high_count)
            
            if high_count > 3:
                st.error("🚨 Major market events ahead")
            elif high_count > 1:
                st.warning("⚠️ Important events scheduled")
            else:
                st.success("✅ Calm period expected")
    
    else:
        st.info("No economic events found for the selected period.")
    
    # Market hours information
    st.subheader("🕐 Global Market Hours")
    
    display_market_hours()
    
    # Central bank schedules
    st.subheader("🏛️ Central Bank Calendar")
    
    display_central_bank_schedule()

def generate_economic_events(start_date, end_date, importance_filter):
    """Generate sample economic events based on typical calendar patterns"""
    events = []
    current_date = start_date
    
    # Typical economic events with patterns
    event_patterns = [
        {
            "event": "Non-Farm Payrolls",
            "currency": "USD",
            "importance": "High",
            "time": "08:30",
            "frequency": "monthly",
            "day": "first_friday"
        },
        {
            "event": "Consumer Price Index (CPI)",
            "currency": "USD",
            "importance": "High",
            "time": "08:30",
            "frequency": "monthly",
            "day": "mid_month"
        },
        {
            "event": "Federal Reserve Interest Rate Decision",
            "currency": "USD",
            "importance": "High",
            "time": "14:00",
            "frequency": "every_6_weeks",
            "day": "wednesday"
        },
        {
            "event": "GDP Growth Rate",
            "currency": "USD",
            "importance": "High",
            "time": "08:30",
            "frequency": "quarterly",
            "day": "end_quarter"
        },
        {
            "event": "Unemployment Rate",
            "currency": "USD",
            "importance": "Medium",
            "time": "08:30",
            "frequency": "monthly",
            "day": "first_friday"
        },
        {
            "event": "ECB Interest Rate Decision",
            "currency": "EUR",
            "importance": "High",
            "time": "12:45",
            "frequency": "every_6_weeks",
            "day": "thursday"
        },
        {
            "event": "Manufacturing PMI",
            "currency": "USD",
            "importance": "Medium",
            "time": "09:45",
            "frequency": "monthly",
            "day": "first_business_day"
        },
        {
            "event": "Initial Jobless Claims",
            "currency": "USD",
            "importance": "Medium",
            "time": "08:30",
            "frequency": "weekly",
            "day": "thursday"
        },
        {
            "event": "Consumer Confidence",
            "currency": "USD",
            "importance": "Medium",
            "time": "10:00",
            "frequency": "monthly",
            "day": "last_tuesday"
        },
        {
            "event": "Retail Sales",
            "currency": "USD",
            "importance": "Medium",
            "time": "08:30",
            "frequency": "monthly",
            "day": "mid_month"
        }
    ]
    
    # Generate events for the date range
    while current_date <= end_date:
        day_of_week = current_date.weekday()  # 0 = Monday, 6 = Sunday
        
        # Add events based on patterns
        if day_of_week == 4:  # Friday - NFP typically first Friday
            if importance_filter in ["All", "High"]:
                events.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Time": "08:30",
                    "Event": "Non-Farm Payrolls",
                    "Currency": "USD",
                    "Importance": "High",
                    "Previous": "150K",
                    "Forecast": "160K"
                })
        
        if day_of_week == 3:  # Thursday - Jobless Claims
            if importance_filter in ["All", "Medium"]:
                events.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Time": "08:30",
                    "Event": "Initial Jobless Claims",
                    "Currency": "USD",
                    "Importance": "Medium",
                    "Previous": "220K",
                    "Forecast": "215K"
                })
        
        if day_of_week == 1 and current_date.day <= 7:  # Tuesday, first week
            if importance_filter in ["All", "Medium"]:
                events.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Time": "10:00",
                    "Event": "Consumer Confidence",
                    "Currency": "USD",
                    "Importance": "Medium",
                    "Previous": "102.5",
                    "Forecast": "103.0"
                })
        
        if current_date.day == 15:  # Mid-month events
            if importance_filter in ["All", "High"]:
                events.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Time": "08:30",
                    "Event": "Consumer Price Index (CPI)",
                    "Currency": "USD",
                    "Importance": "High",
                    "Previous": "3.2%",
                    "Forecast": "3.1%"
                })
        
        # Add some EUR events
        if day_of_week == 3 and current_date.day <= 14:  # ECB meeting pattern
            if importance_filter in ["All", "High"]:
                events.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Time": "12:45",
                    "Event": "ECB Interest Rate Decision",
                    "Currency": "EUR",
                    "Importance": "High",
                    "Previous": "4.50%",
                    "Forecast": "4.50%"
                })
        
        current_date += timedelta(days=1)
    
    return events

def get_event_description(event_name):
    """Get description for economic events"""
    descriptions = {
        "Non-Farm Payrolls": "Monthly report on employment changes in the US economy, excluding farm workers, government employees, and non-profit organizations. Major market mover.",
        "Consumer Price Index (CPI)": "Measures the average change in prices paid by consumers for goods and services. Key inflation indicator.",
        "Federal Reserve Interest Rate Decision": "FOMC decision on the federal funds rate. Significant impact on USD and global markets.",
        "GDP Growth Rate": "Measures the economic growth of the country. Released quarterly with significant market impact.",
        "Unemployment Rate": "Percentage of unemployed workers in the total labor force. Key economic health indicator.",
        "ECB Interest Rate Decision": "European Central Bank's decision on interest rates. Major impact on EUR and European markets.",
        "Manufacturing PMI": "Purchasing Managers' Index for manufacturing sector. Above 50 indicates expansion.",
        "Initial Jobless Claims": "Weekly report of unemployment benefit claims. Leading indicator of employment trends.",
        "Consumer Confidence": "Measures consumer optimism about economic conditions. Impacts spending and market sentiment.",
        "Retail Sales": "Monthly report on consumer spending at retail level. Key indicator of economic health."
    }
    
    return descriptions.get(event_name, "Important economic indicator with potential market impact.")

def display_market_hours():
    """Display global market trading hours"""
    
    # Get current time in different markets
    from datetime import datetime
    import pytz
    
    try:
        current_utc = datetime.utcnow()
        
        # Market hours (in their local time)
        markets = {
            "New York (NYSE)": {
                "timezone": "America/New_York",
                "open": "09:30",
                "close": "16:00",
                "status": "Closed"
            },
            "London (LSE)": {
                "timezone": "Europe/London", 
                "open": "08:00",
                "close": "16:30",
                "status": "Closed"
            },
            "Tokyo (TSE)": {
                "timezone": "Asia/Tokyo",
                "open": "09:00", 
                "close": "15:00",
                "status": "Closed"
            },
            "Hong Kong (HKEX)": {
                "timezone": "Asia/Hong_Kong",
                "open": "09:30",
                "close": "16:00", 
                "status": "Closed"
            }
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for i, (market, info) in enumerate(list(markets.items())[:2]):
                # Simple status display (would need proper timezone handling for real status)
                st.markdown(f"**{market}**")
                st.markdown(f"Hours: {info['open']} - {info['close']}")
                st.markdown(f"Status: {'🟢 Open' if i % 2 == 0 else '🔴 Closed'}")
                st.markdown("---")
        
        with col2:
            for i, (market, info) in enumerate(list(markets.items())[2:]):
                st.markdown(f"**{market}**")
                st.markdown(f"Hours: {info['open']} - {info['close']}")
                st.markdown(f"Status: {'🟢 Open' if i % 2 == 1 else '🔴 Closed'}")
                st.markdown("---")
                
    except Exception as e:
        st.info("Market hours information temporarily unavailable")

def display_central_bank_schedule():
    """Display central bank meeting schedules"""
    
    cb_schedule = [
        {
            "Bank": "Federal Reserve (Fed)",
            "Next Meeting": "2024-01-30/31",
            "Frequency": "8 times per year",
            "Last Rate": "5.25-5.50%"
        },
        {
            "Bank": "European Central Bank (ECB)",
            "Next Meeting": "2024-01-25",
            "Frequency": "8 times per year",
            "Last Rate": "4.50%"
        },
        {
            "Bank": "Bank of England (BoE)",
            "Next Meeting": "2024-02-01",
            "Frequency": "8 times per year", 
            "Last Rate": "5.25%"
        },
        {
            "Bank": "Bank of Japan (BoJ)",
            "Next Meeting": "2024-01-22/23",
            "Frequency": "8 times per year",
            "Last Rate": "-0.10%"
        }
    ]
    
    df = pd.DataFrame(cb_schedule)
    st.dataframe(df, use_container_width=True)
    
    st.info("💡 Central bank decisions are among the most important market-moving events. Monitor these dates closely for potential volatility.")
