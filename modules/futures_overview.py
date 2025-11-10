import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from modules.user_auth import user_auth
from modules import utils
from modules.data_service import get_data_service, format_data_quality_indicator, get_data_quality_color
import json

# Futures contracts and their symbols with fallback ETFs
FUTURES_CONTRACTS = {
    "US Indices": {
        "ES (S&P 500)": {"symbol": "ES=F", "fallback": "SPY", "name": "S&P 500 Futures"},
        "NQ (NASDAQ)": {"symbol": "NQ=F", "fallback": "QQQ", "name": "NASDAQ Futures"},
        "YM (Dow)": {"symbol": "YM=F", "fallback": "DIA", "name": "Dow Futures"},
        "RTY (Russell 2000)": {"symbol": "RTY=F", "fallback": "IWM", "name": "Russell 2000 Futures"}
    },
    "Energy": {
        "CL (Crude Oil)": {"symbol": "CL=F", "fallback": "USO", "name": "Crude Oil Futures"},
        "NG (Natural Gas)": {"symbol": "NG=F", "fallback": "UNG", "name": "Natural Gas Futures"},
        "HO (Heating Oil)": {"symbol": "HO=F", "fallback": "UHN", "name": "Heating Oil Futures"},
        "RB (Gasoline)": {"symbol": "RB=F", "fallback": "UGA", "name": "Gasoline Futures"}
    },
    "Metals": {
        "GC (Gold)": {"symbol": "GC=F", "fallback": "GLD", "name": "Gold Futures"},
        "SI (Silver)": {"symbol": "SI=F", "fallback": "SLV", "name": "Silver Futures"},
        "HG (Copper)": {"symbol": "HG=F", "fallback": "CPER", "name": "Copper Futures"},
        "PA (Palladium)": {"symbol": "PA=F", "fallback": "PALL", "name": "Palladium Futures"},
        "PL (Platinum)": {"symbol": "PL=F", "fallback": "PPLT", "name": "Platinum Futures"}
    },
    "Agriculture": {
        "ZW (Wheat)": {"symbol": "ZW=F", "fallback": "WEAT", "name": "Wheat Futures"},
        "ZC (Corn)": {"symbol": "ZC=F", "fallback": "CORN", "name": "Corn Futures"},
        "ZS (Soybeans)": {"symbol": "ZS=F", "fallback": "SOYB", "name": "Soybean Futures"},
        "CT (Cotton)": {"symbol": "CT=F", "fallback": "BAL", "name": "Cotton Futures"},
        "SB (Sugar)": {"symbol": "SB=F", "fallback": "CANE", "name": "Sugar Futures"}
    },
    "Currency": {
        "EUR/USD": {"symbol": "EURUSD=X", "fallback": "FXE", "name": "Euro/USD"},
        "GBP/USD": {"symbol": "GBPUSD=X", "fallback": "FXB", "name": "British Pound/USD"},
        "USD/JPY": {"symbol": "USDJPY=X", "fallback": "FXY", "name": "USD/Japanese Yen"},
        "USD/CAD": {"symbol": "USDCAD=X", "fallback": "FXC", "name": "USD/Canadian Dollar"}
    },
    "Canadian": {
        "TSX Index": {"symbol": "^GSPTSE", "fallback": "TDB902", "name": "TSX Composite Index"},
        "CAD Futures": {"symbol": "CAD=X", "fallback": "FXC", "name": "Canadian Dollar Futures"}
    }
}

def get_futures_data(symbol, fallback_symbol=None, period="1d", force_refresh=False):
    """Get futures data with real-time capabilities and fallback support"""
    try:
        data_service = get_data_service()
        
        # For real-time data (current prices), use data service
        if period == "1d":
            # Try to get real-time quote first
            quote_data = data_service.get_stock_quote(symbol, force_refresh=force_refresh)
            
            # If primary symbol fails and we have fallback, try fallback
            if not quote_data and fallback_symbol:
                quote_data = data_service.get_stock_quote(fallback_symbol, force_refresh=force_refresh)
                symbol = fallback_symbol  # Update symbol for consistency
            
            if quote_data:
                # Get historical data for charting
                hist = data_service.get_historical_data(symbol, period=period)
                if hist is None:
                    # Fallback to yfinance for historical data if needed
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period, interval="1m")
                
                # Get company info
                info = data_service.get_company_info(symbol) or {}
                
                result = {
                    'symbol': symbol,
                    'history': hist if hist is not None else pd.DataFrame(),
                    'current_price': quote_data.get('price', 0),
                    'change': quote_data.get('change', 0),
                    'change_percent': quote_data.get('change_percent', 0),
                    'volume': quote_data.get('volume', 0),
                    'info': info,
                    'data_quality': quote_data.get('data_quality', 'unknown'),
                    'source': quote_data.get('source', 'unknown'),
                    'timestamp': quote_data.get('timestamp', datetime.now().isoformat()),
                    'cache_age': quote_data.get('cache_age', 0),
                    'quality_indicator': format_data_quality_indicator(
                        quote_data.get('data_quality', 'unknown'), 
                        quote_data.get('cache_age', 0)
                    ),
                    'quality_color': get_data_quality_color(quote_data.get('data_quality', 'unknown'))
                }
                return result
        
        # Fallback to original yfinance method for longer periods or if real-time fails
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1m" if period == "1d" else "1d")
        
        if hist.empty and fallback_symbol:
            # Try fallback symbol
            ticker = yf.Ticker(fallback_symbol)
            hist = ticker.history(period=period, interval="1m" if period == "1d" else "1d")
            symbol = fallback_symbol  # Update symbol for info retrieval
        
        if not hist.empty:
            info = ticker.info
            current_price = hist['Close'].iloc[-1]
            
            # Calculate change
            if len(hist) >= 2:
                prev_price = hist['Close'].iloc[-2] if period == "1d" else hist['Close'].iloc[0]
                change = current_price - prev_price
                change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
            else:
                change = 0
                change_percent = 0
            
            return {
                'symbol': symbol,
                'history': hist,
                'current_price': current_price,
                'change': change,
                'change_percent': change_percent,
                'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                'info': info,
                'data_quality': 'fallback',
                'source': 'yfinance',
                'timestamp': datetime.now().isoformat(),
                'cache_age': 0,
                'quality_indicator': '🔴 DELAYED',
                'quality_color': 'red'
            }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

def create_mini_chart(data, title, width=300, height=200):
    """Create a mini chart for finviz-style display"""
    if not data or data['history'].empty:
        return None
    
    hist = data['history']
    change_color = 'green' if data['change'] >= 0 else 'red'
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        mode='lines',
        name='Price',
        line=dict(color=change_color, width=2),
        hovertemplate='<b>%{y:.2f}</b><br>%{x}<extra></extra>'
    ))
    
    # Update layout for compact display
    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='color:{change_color}'>${data['current_price']:.2f} ({data['change']:+.2f}, {data['change_percent']:+.2f}%)</span>",
            font=dict(size=12, color='white'),
            x=0.5
        ),
        template="plotly_dark",
        width=width,
        height=height,
        margin=dict(l=30, r=30, t=60, b=30),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showticklabels=True,
            zeroline=False,
            tickformat='.2f'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x'
    )
    
    return fig

def get_market_status():
    """Get current market status"""
    now = datetime.now()
    
    # Simplified market hours (US Eastern Time approximation)
    if now.weekday() < 5:  # Monday to Friday
        if 9 <= now.hour < 16:
            return "🟢 Market Open"
        elif 16 <= now.hour < 20 or 4 <= now.hour < 9:
            return "🟡 Extended Hours"
        else:
            return "🔴 Market Closed"
    else:
        return "🔴 Market Closed (Weekend)"

def display_category_section(category_name, contracts, period, favorites, show_category):
    """Display a category section with futures charts"""
    
    with st.expander(f"📊 {category_name}", expanded=show_category):
        if show_category:
            # Create grid layout - 3 charts per row for optimal viewing
            charts_per_row = 3
            contract_items = list(contracts.items())
            
            for i in range(0, len(contract_items), charts_per_row):
                cols = st.columns(charts_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(contract_items):
                        contract_name, contract_info = contract_items[i + j]
                        
                        with col:
                            # Get data for this contract
                            data = get_futures_data(
                                contract_info['symbol'], 
                                contract_info['fallback'], 
                                period
                            )
                            
                            if data:
                                # Create mini chart
                                chart = create_mini_chart(data, contract_name)
                                
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True, key=f"chart_{category_name}_{contract_name}")
                                    
                                    # Add to favorites button
                                    is_favorite = contract_info['symbol'] in favorites
                                    fav_text = "⭐ Remove from Favorites" if is_favorite else "☆ Add to Favorites"
                                    
                                    if st.button(fav_text, key=f"fav_{category_name}_{contract_name}"):
                                        if is_favorite:
                                            favorites.remove(contract_info['symbol'])
                                        else:
                                            favorites.append(contract_info['symbol'])
                                        
                                        # Save to user preferences
                                        user_auth.auto_save_preference('futures', 'favorites', favorites)
                                        st.rerun()
                            else:
                                st.error(f"Unable to load data for {contract_name}")

def get_market_movers(period="1d"):
    """Get biggest movers in futures markets"""
    movers_data = []
    
    # Collect data from all contracts
    for category, contracts in FUTURES_CONTRACTS.items():
        for contract_name, contract_info in contracts.items():
            data = get_futures_data(contract_info['symbol'], contract_info['fallback'], period)
            if data:
                movers_data.append({
                    'Contract': contract_name,
                    'Category': category,
                    'Price': data['current_price'],
                    'Change': data['change'],
                    'Change %': data['change_percent'],
                    'Volume': data['volume']
                })
    
    if movers_data:
        df = pd.DataFrame(movers_data)
        
        # Sort by absolute percentage change for biggest movers
        df['Abs_Change'] = abs(df['Change %'])
        biggest_movers = df.nlargest(10, 'Abs_Change')
        
        # Top gainers and losers
        gainers = df.nlargest(5, 'Change %')
        losers = df.nsmallest(5, 'Change %')
        
        # Volume leaders
        volume_leaders = df.nlargest(5, 'Volume')
        
        return {
            'biggest_movers': biggest_movers,
            'gainers': gainers,
            'losers': losers,
            'volume_leaders': volume_leaders
        }
    
    return None

def display_market_summary():
    """Display market summary section"""
    st.subheader("📈 Market Summary")
    
    movers = get_market_movers()
    
    if movers:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**🚀 Top Gainers**")
            for _, row in movers['gainers'].head(3).iterrows():
                color = "green" if row['Change %'] > 0 else "red"
                st.markdown(f"<span style='color:{color}'>{row['Contract']}: {row['Change %']:+.2f}%</span>", 
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📉 Top Losers**")
            for _, row in movers['losers'].head(3).iterrows():
                color = "green" if row['Change %'] > 0 else "red"
                st.markdown(f"<span style='color:{color}'>{row['Contract']}: {row['Change %']:+.2f}%</span>", 
                           unsafe_allow_html=True)
        
        with col3:
            st.markdown("**📊 Biggest Movers**")
            for _, row in movers['biggest_movers'].head(3).iterrows():
                color = "green" if row['Change %'] > 0 else "red"
                st.markdown(f"<span style='color:{color}'>{row['Contract']}: {abs(row['Change %']):.2f}%</span>", 
                           unsafe_allow_html=True)
        
        with col4:
            st.markdown("**🔊 Volume Leaders**")
            for _, row in movers['volume_leaders'].head(3).iterrows():
                if row['Volume'] > 0:
                    st.markdown(f"{row['Contract']}: {utils.format_volume(row['Volume'])}")

def display_favorites_section(favorites, period):
    """Display favorites section"""
    if favorites:
        st.subheader("⭐ My Favorites")
        
        # Create grid for favorites
        charts_per_row = 4
        for i in range(0, len(favorites), charts_per_row):
            cols = st.columns(charts_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(favorites):
                    favorite_symbol = favorites[i + j]
                    
                    # Find contract info
                    contract_name = None
                    contract_info = None
                    
                    for category, contracts in FUTURES_CONTRACTS.items():
                        for name, info in contracts.items():
                            if info['symbol'] == favorite_symbol:
                                contract_name = name
                                contract_info = info
                                break
                        if contract_name:
                            break
                    
                    if contract_info:
                        with col:
                            data = get_futures_data(
                                contract_info['symbol'], 
                                contract_info['fallback'], 
                                period
                            )
                            
                            if data:
                                chart = create_mini_chart(data, contract_name)
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True, key=f"fav_chart_{favorite_symbol}")

def display_futures_overview():
    """Main display function for futures overview"""
    
    # Header with market status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("🔮 Futures Overview")
    with col2:
        st.info(get_market_status())
    
    # Get user preferences
    favorites = user_auth.get_user_preference('futures', 'favorites', [])
    expanded_categories = user_auth.get_user_preference('futures', 'expanded_categories', 
                                                       {cat: True for cat in FUTURES_CONTRACTS.keys()})
    default_period = user_auth.get_user_preference('futures', 'default_period', '1d')
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**📊 Finviz-Style Futures Dashboard**")
        st.markdown("*Real-time futures data with Canadian and American market coverage*")
    
    with col2:
        period = st.selectbox(
            "Time Period", 
            ["1d", "5d", "1mo"], 
            index=["1d", "5d", "1mo"].index(default_period),
            key="futures_period"
        )
        # Save preference
        user_auth.auto_save_preference('futures', 'default_period', period)
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Display favorites first if any
    if favorites:
        display_favorites_section(favorites, period)
        st.markdown("---")
    
    # Market summary
    display_market_summary()
    st.markdown("---")
    
    # Category selection
    st.subheader("📋 Category Settings")
    category_cols = st.columns(len(FUTURES_CONTRACTS))
    
    for i, category in enumerate(FUTURES_CONTRACTS.keys()):
        with category_cols[i]:
            show_category = st.checkbox(
                f"Show {category}", 
                value=expanded_categories.get(category, True),
                key=f"show_{category}"
            )
            expanded_categories[category] = show_category
    
    # Save category preferences
    user_auth.auto_save_preference('futures', 'expanded_categories', expanded_categories)
    
    st.markdown("---")
    
    # Display each category
    for category_name, contracts in FUTURES_CONTRACTS.items():
        if expanded_categories.get(category_name, True):
            display_category_section(category_name, contracts, period, favorites, True)
    
    # Footer with disclaimers
    st.markdown("---")
    st.markdown("""
    **📊 Data Sources & Notes:**
    - Primary: Yahoo Finance futures data
    - Fallback: Related ETFs when futures data unavailable
    - Updates: Real-time with 1-minute cache
    - Coverage: US, Canadian, Energy, Metals, Agriculture, Currency futures
    
    **⚠️ Disclaimer:** For educational purposes only. Not financial advice.
    """)