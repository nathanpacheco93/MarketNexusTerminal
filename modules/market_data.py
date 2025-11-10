import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from modules.data_service import get_data_service, format_data_quality_indicator, get_data_quality_color

def get_index_data(symbol, show_quality_indicator=True):
    """Get real-time data for major indices with quality indicators"""
    try:
        data_service = get_data_service()
        data = data_service.get_index_data(symbol)
        
        if data:
            result = {
                'price': data.get('price'),
                'change': data.get('change'),
                'change_percent': data.get('change_percent'),
                'volume': data.get('volume', 0),
                'data_quality': data.get('data_quality'),
                'source': data.get('source'),
                'timestamp': data.get('timestamp'),
                'cache_age': data.get('cache_age', 0)
            }
            
            if show_quality_indicator:
                quality_indicator = format_data_quality_indicator(
                    data.get('data_quality'), data.get('cache_age', 0)
                )
                result['quality_indicator'] = quality_indicator
                result['quality_color'] = get_data_quality_color(data.get('data_quality'))
            
            return result
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

def get_currency_data(symbol, show_quality_indicator=True):
    """Get real-time currency data with quality indicators"""
    try:
        data_service = get_data_service()
        data = data_service.get_currency_data(symbol)
        
        if data:
            result = {
                'price': data.get('price'),
                'change': data.get('change'),
                'change_percent': data.get('change_percent', 0),
                'data_quality': data.get('data_quality'),
                'source': data.get('source'),
                'timestamp': data.get('timestamp'),
                'cache_age': data.get('cache_age', 0)
            }
            
            if show_quality_indicator:
                quality_indicator = format_data_quality_indicator(
                    data.get('data_quality'), data.get('cache_age', 0)
                )
                result['quality_indicator'] = quality_indicator
                result['quality_color'] = get_data_quality_color(data.get('data_quality'))
            
            return result
    except Exception as e:
        st.error(f"Error fetching currency data for {symbol}: {str(e)}")
    return None

def get_commodity_data(symbol, show_quality_indicator=True):
    """Get real-time commodity data with quality indicators"""
    try:
        data_service = get_data_service()
        data = data_service.get_commodity_data(symbol)
        
        if data:
            result = {
                'price': data.get('price'),
                'change': data.get('change'),
                'change_percent': data.get('change_percent', 0),
                'data_quality': data.get('data_quality'),
                'source': data.get('source'),
                'timestamp': data.get('timestamp'),
                'cache_age': data.get('cache_age', 0)
            }
            
            if show_quality_indicator:
                quality_indicator = format_data_quality_indicator(
                    data.get('data_quality'), data.get('cache_age', 0)
                )
                result['quality_indicator'] = quality_indicator
                result['quality_color'] = get_data_quality_color(data.get('data_quality'))
            
            return result
    except Exception as e:
        st.error(f"Error fetching commodity data for {symbol}: {str(e)}")
    return None

def get_crypto_data(symbol, show_quality_indicator=True):
    """Get real-time cryptocurrency data with quality indicators"""
    try:
        data_service = get_data_service()
        data = data_service.get_crypto_data(symbol)
        
        if data:
            result = {
                'price': data.get('price'),
                'change': data.get('change'),
                'change_percent': data.get('change_percent', 0),
                'data_quality': data.get('data_quality'),
                'source': data.get('source'),
                'timestamp': data.get('timestamp'),
                'cache_age': data.get('cache_age', 0)
            }
            
            if show_quality_indicator:
                quality_indicator = format_data_quality_indicator(
                    data.get('data_quality'), data.get('cache_age', 0)
                )
                result['quality_indicator'] = quality_indicator
                result['quality_color'] = get_data_quality_color(data.get('data_quality'))
            
            return result
    except Exception as e:
        st.error(f"Error fetching crypto data for {symbol}: {str(e)}")
    return None

@st.cache_data(ttl=300)  # Cache for 5 minutes for historical data
def get_stock_data(symbol, period="1y"):
    """Get historical stock data with company info"""
    try:
        data_service = get_data_service()
        
        # Get historical data
        hist = data_service.get_historical_data(symbol, period=period)
        
        # Get company info (cached separately for longer)
        info = data_service.get_company_info(symbol)
        
        if hist is not None:
            result = {
                'history': hist,
                'info': info or {},
                'data_quality': 'historical',
                'source': 'yfinance',
                'timestamp': datetime.now().isoformat()
            }
            return result
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {str(e)}")
    return None

def get_real_time_quote(symbol, force_refresh=False, show_quality_indicator=True):
    """Get real-time quote for a symbol with enhanced capabilities"""
    try:
        data_service = get_data_service()
        data = data_service.get_stock_quote(symbol, force_refresh=force_refresh)
        
        if data:
            result = {
                'symbol': symbol,
                'price': data.get('price'),
                'change': data.get('change'),
                'change_percent': data.get('change_percent'),
                'volume': data.get('volume', 0),
                'timestamp': data.get('timestamp'),
                'data_quality': data.get('data_quality'),
                'source': data.get('source'),
                'cache_age': data.get('cache_age', 0)
            }
            
            if show_quality_indicator:
                quality_indicator = format_data_quality_indicator(
                    data.get('data_quality'), data.get('cache_age', 0)
                )
                result['quality_indicator'] = quality_indicator
                result['quality_color'] = get_data_quality_color(data.get('data_quality'))
            
            return result
    except Exception as e:
        st.error(f"Error fetching real-time quote for {symbol}: {str(e)}")
    return None

def get_multiple_quotes(symbols, force_refresh=False, show_quality_indicator=True):
    """Get multiple real-time quotes efficiently"""
    try:
        data_service = get_data_service()
        quotes = data_service.get_multiple_quotes(symbols, force_refresh=force_refresh)
        
        results = {}
        for symbol, data in quotes.items():
            if data:
                result = {
                    'symbol': symbol,
                    'price': data.get('price'),
                    'change': data.get('change'),
                    'change_percent': data.get('change_percent'),
                    'volume': data.get('volume', 0),
                    'timestamp': data.get('timestamp'),
                    'data_quality': data.get('data_quality'),
                    'source': data.get('source'),
                    'cache_age': data.get('cache_age', 0)
                }
                
                if show_quality_indicator:
                    quality_indicator = format_data_quality_indicator(
                        data.get('data_quality'), data.get('cache_age', 0)
                    )
                    result['quality_indicator'] = quality_indicator
                    result['quality_color'] = get_data_quality_color(data.get('data_quality'))
                
                results[symbol] = result
        
        return results
    except Exception as e:
        st.error(f"Error fetching multiple quotes: {str(e)}")
    return {}

def display_data_quality_indicator(data, label="Data Quality"):
    """Display data quality indicator in Streamlit"""
    if data and 'quality_indicator' in data:
        quality_color = data.get('quality_color', 'gray')
        st.markdown(
            f"**{label}:** <span style='color: {quality_color}'>{data['quality_indicator']}</span>",
            unsafe_allow_html=True
        )
        
        # Show additional info on hover/expand
        if data.get('source'):
            st.caption(f"Source: {data['source']}")
        if data.get('timestamp'):
            try:
                from datetime import datetime
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                st.caption(f"Last updated: {timestamp.strftime('%H:%M:%S')}")
            except:
                pass

def display_auto_refresh_controls():
    """Display auto-refresh controls for real-time data"""
    st.markdown("### 🔄 Real-Time Data Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=False, help="Automatically refresh data every 5 seconds")
    
    with col2:
        if st.button("🔄 Refresh Now", help="Force refresh all data"):
            # Clear relevant caches to force refresh
            data_service = get_data_service()
            data_service.clear_cache('stock_quote')
            data_service.clear_cache('index_data')
            st.success("Data refreshed!")
            st.rerun()
    
    with col3:
        market_hours = is_market_hours()
        market_status = "🟢 OPEN" if market_hours else "🔴 CLOSED"
        st.markdown(f"**Market:** {market_status}")
    
    # Auto-refresh implementation
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()
    
    return auto_refresh

def is_market_hours():
    """Check if it's currently market hours"""
    try:
        data_service = get_data_service()
        return data_service.is_market_hours()
    except:
        return False

def get_cache_statistics():
    """Get and display cache statistics"""
    try:
        data_service = get_data_service()
        stats = data_service.get_cache_stats()
        
        st.markdown("### 📊 Cache Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Entries", stats['total_entries'])
        
        with col2:
            size_mb = stats['total_size_bytes'] / (1024 * 1024)
            st.metric("Cache Size", f"{size_mb:.2f} MB")
        
        with col3:
            if st.button("🗑️ Clear Cache"):
                data_service.clear_cache()
                st.success("Cache cleared!")
                st.rerun()
        
        # Show cache breakdown by type
        if stats['type_counts']:
            st.markdown("**Cache Breakdown:**")
            for data_type, count in stats['type_counts'].items():
                st.write(f"- {data_type}: {count} entries")
                
    except Exception as e:
        st.error(f"Error getting cache statistics: {str(e)}")

# Backward compatibility functions
def get_stock_quote_simple(symbol):
    """Simple stock quote function for backward compatibility"""
    data = get_real_time_quote(symbol, show_quality_indicator=False)
    if data:
        return {
            'price': data.get('price'),
            'change': data.get('change'),
            'change_percent': data.get('change_percent'),
            'volume': data.get('volume', 0)
        }
    return None

def format_price_display(price, change, change_percent):
    """Format price display with color coding"""
    if change is None or change_percent is None:
        return f"${price:.2f}"
    
    color = "green" if change >= 0 else "red"
    sign = "+" if change >= 0 else ""
    
    return f"${price:.2f} <span style='color: {color}'>{sign}{change:.2f} ({sign}{change_percent:.2f}%)</span>"
