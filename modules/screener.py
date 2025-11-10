import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modules.user_auth import UserAuth

def display_screener():
    """Display market screener section"""
    
    st.subheader("🔍 Stock Screener")
    
    # Screening criteria - expanded to 4 columns for volume criteria
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Price Criteria**")
        # Load user preferences with defaults
        min_price = st.number_input(
            "Min Price ($)", 
            min_value=0.0, 
            value=UserAuth.get_user_preference('screener', 'min_price', 0.0)
        )
        max_price = st.number_input(
            "Max Price ($)", 
            min_value=0.0, 
            value=UserAuth.get_user_preference('screener', 'max_price', 1000.0)
        )
    
    with col2:
        st.markdown("**Market Cap**")
        market_cap_filter = st.selectbox(
            "Market Cap Range",
            ["All", "Large Cap (>$10B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)"],
            index=0 if not UserAuth.get_user_preference('screener', 'market_cap_filter', None) else 
            ["All", "Large Cap (>$10B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)"].index(UserAuth.get_user_preference('screener', 'market_cap_filter', "All"))
        )
    
    with col3:
        st.markdown("**Performance**")
        performance_filter = st.selectbox(
            "Performance Filter",
            ["All", "Top Gainers", "Top Losers", "High Volume"],
            index=0 if not UserAuth.get_user_preference('screener', 'performance_filter', None) else
            ["All", "Top Gainers", "Top Losers", "High Volume"].index(UserAuth.get_user_preference('screener', 'performance_filter', "All"))
        )
    
    with col4:
        st.markdown("**Volume Criteria**")
        min_volume = st.number_input(
            "Min Volume Threshold",
            min_value=0,
            value=UserAuth.get_user_preference('screener', 'min_volume', 0),
            step=100000,
            help="Minimum daily volume required"
        )
        
        volume_comparison = st.selectbox(
            "Volume vs Average",
            ["Any", "Above 20-day Average", "Below 20-day Average", "Above 3-month Average", "Below 3-month Average"],
            index=0 if not UserAuth.get_user_preference('screener', 'volume_comparison', None) else
            ["Any", "Above 20-day Average", "Below 20-day Average", "Above 3-month Average", "Below 3-month Average"].index(UserAuth.get_user_preference('screener', 'volume_comparison', "Any"))
        )
        
        volume_ratio_filter = st.selectbox(
            "Volume Ratio Filter",
            ["Any", "Above Ratio Threshold", "Below Ratio Threshold"],
            index=0 if not UserAuth.get_user_preference('screener', 'volume_ratio_filter', None) else
            ["Any", "Above Ratio Threshold", "Below Ratio Threshold"].index(UserAuth.get_user_preference('screener', 'volume_ratio_filter', "Any"))
        )
        
        volume_ratio_threshold = st.number_input(
            "Volume Ratio (x Average)",
            min_value=0.1,
            max_value=10.0,
            value=UserAuth.get_user_preference('screener', 'volume_ratio_threshold', 2.0),
            step=0.1,
            help="Multiplier vs 20-day average (e.g., 2.0 = 2x average volume)"
        )
    
    # Sector filter
    sectors = [
        "All Sectors",
        "Technology",
        "Healthcare", 
        "Financial Services",
        "Consumer Cyclical",
        "Industrials",
        "Communication Services",
        "Consumer Defensive",
        "Energy",
        "Real Estate",
        "Materials",
        "Utilities"
    ]
    
    selected_sector = st.selectbox("Sector Filter", sectors)
    
    # Auto-save user preferences when values change
    if UserAuth.is_logged_in():
        UserAuth.auto_save_preference('screener', 'min_price', min_price)
        UserAuth.auto_save_preference('screener', 'max_price', max_price)
        UserAuth.auto_save_preference('screener', 'market_cap_filter', market_cap_filter)
        UserAuth.auto_save_preference('screener', 'performance_filter', performance_filter)
        UserAuth.auto_save_preference('screener', 'min_volume', min_volume)
        UserAuth.auto_save_preference('screener', 'volume_comparison', volume_comparison)
        UserAuth.auto_save_preference('screener', 'volume_ratio_filter', volume_ratio_filter)
        UserAuth.auto_save_preference('screener', 'volume_ratio_threshold', volume_ratio_threshold)
    
    # Screen button
    if st.button("🔍 Run Screen"):
        with st.spinner("Screening stocks..."):
            screened_stocks = run_stock_screen(
                min_price, max_price, market_cap_filter, 
                performance_filter, selected_sector,
                min_volume, volume_comparison, volume_ratio_filter, volume_ratio_threshold
            )
            
            if screened_stocks:
                st.success(f"Found {len(screened_stocks)} stocks matching your criteria")
                
                # Display results
                df = pd.DataFrame(screened_stocks)
                
                # Sort by market cap or performance based on filter
                if performance_filter == "Top Gainers":
                    df = df.sort_values("Change %", ascending=False)
                elif performance_filter == "Top Losers":
                    df = df.sort_values("Change %", ascending=True)
                elif performance_filter == "High Volume":
                    df = df.sort_values("Current Volume", ascending=False)
                elif volume_comparison != "Any" or volume_ratio_filter != "Any":
                    # Sort by volume ratio when volume filters are applied
                    df = df.sort_values("Volume Ratio (20d)", ascending=False)
                else:
                    df = df.sort_values("Market Cap ($B)", ascending=False)
                
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"screen_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("No stocks found matching your criteria. Try adjusting the filters.")
    
    # Predefined screens
    st.subheader("📋 Predefined Screens")
    
    # Traditional screens
    st.markdown("**Traditional Screens:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💎 Value Stocks"):
            run_predefined_screen("value")
    
    with col2:
        if st.button("🚀 Growth Stocks"):
            run_predefined_screen("growth")
    
    with col3:
        if st.button("💰 Dividend Stocks"):
            run_predefined_screen("dividend")
    
    with col4:
        if st.button("📈 Momentum Stocks"):
            run_predefined_screen("momentum")
    
    # Volume-based screens
    st.markdown("**Volume-Based Screens:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔥 Volume Breakouts"):
            run_predefined_screen("volume_breakouts")
    
    with col2:
        if st.button("🔍 Low Volume Consolidation"):
            run_predefined_screen("low_volume_consolidation")
    
    with col3:
        if st.button("⚡ Volume Surge Detection"):
            run_predefined_screen("volume_surge")
    
    # Popular stock lists
    st.subheader("📊 Popular Lists")
    
    popular_lists = {
        "S&P 500": get_sp500_symbols(),
        "NASDAQ 100": get_nasdaq100_symbols(),
        "DOW 30": get_dow30_symbols(),
        "Tech Giants": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"],
        "Banking": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PXD"]
    }
    
    selected_list = st.selectbox("View Popular List", list(popular_lists.keys()))
    
    if st.button("📋 Load List"):
        symbols = popular_lists[selected_list]
        display_stock_list(symbols, selected_list)

def run_stock_screen(min_price, max_price, market_cap_filter, performance_filter, sector_filter, min_volume=0, volume_comparison="Any", volume_ratio_filter="Any", volume_ratio_threshold=2.0):
    """Run stock screening with given criteria including volume filters"""
    try:
        # Get a sample of popular stocks to screen
        # In a real application, you would screen from a larger universe
        sample_symbols = get_sample_symbols()
        
        screened_stocks = []
        
        for symbol in sample_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if hist.empty or len(hist) < 2:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change = current_price - prev_price
                change_percent = (change / prev_price) * 100
                
                # Price filter
                if current_price < min_price or current_price > max_price:
                    continue
                
                # Market cap filter
                market_cap = info.get('marketCap', 0)
                if market_cap_filter == "Large Cap (>$10B)" and market_cap < 10e9:
                    continue
                elif market_cap_filter == "Mid Cap ($2B-$10B)" and (market_cap < 2e9 or market_cap > 10e9):
                    continue
                elif market_cap_filter == "Small Cap (<$2B)" and market_cap > 2e9:
                    continue
                
                # Sector filter
                sector = info.get('sector', '')
                if sector_filter != "All Sectors" and sector_filter.lower() not in sector.lower():
                    continue
                
                # Performance filter
                volume = hist['Volume'].iloc[-1]
                if performance_filter == "Top Gainers" and change_percent < 2:
                    continue
                elif performance_filter == "Top Losers" and change_percent > -2:
                    continue
                elif performance_filter == "High Volume" and volume < 1000000:
                    continue
                
                # Calculate volume metrics
                volume_metrics = calculate_volume_metrics(symbol)
                
                # Apply volume filters
                if not check_volume_filters(volume_metrics, min_volume, volume_comparison, volume_ratio_filter, volume_ratio_threshold):
                    continue
                
                # Prepare volume display data
                if volume_metrics:
                    volume_display = format_volume(volume_metrics['current_volume'])
                    avg_volume_20d_display = format_volume(volume_metrics['avg_volume_20d'])
                    avg_volume_3m_display = format_volume(volume_metrics['avg_volume_3m'])
                    volume_ratio_20d_display = f"{volume_metrics['volume_ratio_20d']:.2f}x"
                    volume_ratio_3m_display = f"{volume_metrics['volume_ratio_3m']:.2f}x"
                else:
                    volume_display = format_volume(volume)
                    avg_volume_20d_display = "N/A"
                    avg_volume_3m_display = "N/A"
                    volume_ratio_20d_display = "N/A"
                    volume_ratio_3m_display = "N/A"
                
                screened_stocks.append({
                    'Symbol': symbol,
                    'Company': info.get('longName', symbol),
                    'Price': f"${current_price:.2f}",
                    'Change': f"${change:.2f}",
                    'Change %': f"{change_percent:.2f}%",
                    'Current Volume': volume_display,
                    'Avg Volume (20d)': avg_volume_20d_display,
                    'Avg Volume (3m)': avg_volume_3m_display,
                    'Volume Ratio (20d)': volume_ratio_20d_display,
                    'Volume Ratio (3m)': volume_ratio_3m_display,
                    'Market Cap ($B)': f"{market_cap/1e9:.2f}" if market_cap else "N/A",
                    'Sector': sector,
                    'P/E Ratio': f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A"
                })
                
            except Exception as e:
                continue
        
        return screened_stocks
        
    except Exception as e:
        st.error(f"Error running screen: {str(e)}")
        return []

def run_predefined_screen(screen_type):
    """Run predefined screening strategies"""
    st.info(f"Running {screen_type.replace('_', ' ').title()} screen...")
    
    if screen_type == "value":
        # Value stocks - low P/E, reasonable price, decent volume
        screened = run_stock_screen(5, 500, "All", "All", "All Sectors", min_volume=500000)
        # Filter for low P/E ratios
        value_stocks = []
        for stock in screened:
            try:
                pe = float(stock['P/E Ratio'].replace('N/A', '999'))
                if pe < 20 and pe > 0:
                    value_stocks.append(stock)
            except:
                continue
        
        if value_stocks:
            st.success(f"Found {len(value_stocks)} value stocks with decent volume")
            df = pd.DataFrame(value_stocks)
            df = df.sort_values("P/E Ratio")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No value stocks found in the current sample")
    
    elif screen_type == "growth":
        # Growth stocks - technology sector focus, above average volume
        screened = run_stock_screen(10, 1000, "All", "All", "Technology", min_volume=1000000, volume_comparison="Above 20-day Average")
        if screened:
            st.success(f"Found {len(screened)} technology growth stocks with strong volume")
            df = pd.DataFrame(screened)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No growth stocks found")
    
    elif screen_type == "dividend":
        # Dividend stocks - typically utilities, REITs, steady volume
        screened = run_stock_screen(5, 200, "All", "All", "Utilities", min_volume=250000)
        if screened:
            st.success(f"Found {len(screened)} potential dividend stocks with steady volume")
            df = pd.DataFrame(screened)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No dividend stocks found")
    
    elif screen_type == "momentum":
        # Momentum stocks - top gainers with high volume
        screened = run_stock_screen(5, 1000, "All", "Top Gainers", "All Sectors", min_volume=2000000, volume_comparison="Above 20-day Average")
        if screened:
            st.success(f"Found {len(screened)} momentum stocks with strong volume confirmation")
            df = pd.DataFrame(screened)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No momentum stocks found")
    
    elif screen_type == "volume_breakouts":
        # High volume breakouts - stocks with 3x+ average volume and positive price movement
        screened = run_stock_screen(1, 1000, "All", "Top Gainers", "All Sectors", min_volume=1000000, volume_ratio_filter="Above Ratio Threshold", volume_ratio_threshold=3.0)
        if screened:
            st.success(f"Found {len(screened)} high volume breakout candidates")
            st.info("📈 These stocks show strong volume surges (3x+ average) with positive price movement - potential breakout patterns")
            df = pd.DataFrame(screened)
            df = df.sort_values("Volume Ratio (20d)", ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No volume breakouts found in current market conditions")
    
    elif screen_type == "low_volume_consolidation":
        # Low volume consolidation - stocks with below average volume and minimal price movement
        screened = run_stock_screen(5, 1000, "All", "All", "All Sectors", min_volume=100000, volume_comparison="Below 20-day Average", volume_ratio_filter="Below Ratio Threshold", volume_ratio_threshold=0.7)
        
        # Filter for minimal price movement (consolidation pattern)
        consolidation_stocks = []
        for stock in screened:
            try:
                change_percent = float(stock['Change %'].replace('%', ''))
                if abs(change_percent) < 3:  # Less than 3% movement
                    consolidation_stocks.append(stock)
            except:
                continue
        
        if consolidation_stocks:
            st.success(f"Found {len(consolidation_stocks)} low volume consolidation patterns")
            st.info("🔍 These stocks show consolidation with low volume - potential for future moves when volume returns")
            df = pd.DataFrame(consolidation_stocks)
            df = df.sort_values("Volume Ratio (20d)", ascending=True)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No consolidation patterns found")
    
    elif screen_type == "volume_surge":
        # Volume surge detection - stocks with 2x+ volume but any price movement
        screened = run_stock_screen(1, 1000, "All", "All", "All Sectors", min_volume=500000, volume_ratio_filter="Above Ratio Threshold", volume_ratio_threshold=2.0)
        if screened:
            st.success(f"Found {len(screened)} stocks with significant volume surges")
            st.info("⚡ These stocks show 2x+ average volume - indicates institutional interest or news flow")
            df = pd.DataFrame(screened)
            df = df.sort_values("Volume Ratio (20d)", ascending=False)
            st.dataframe(df, use_container_width=True)
            
            # Show volume surge statistics
            try:
                avg_ratio = df['Volume Ratio (20d)'].str.replace('x', '').astype(float).mean()
                max_ratio = df['Volume Ratio (20d)'].str.replace('x', '').astype(float).max()
                st.metric("Average Volume Ratio", f"{avg_ratio:.2f}x")
                st.metric("Max Volume Ratio", f"{max_ratio:.2f}x")
            except:
                pass
        else:
            st.warning("No significant volume surges detected")

def display_stock_list(symbols, list_name):
    """Display a list of stocks"""
    st.info(f"Loading {list_name} stocks...")
    
    stock_data = []
    for symbol in symbols[:20]:  # Limit to first 20 to avoid timeouts
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                
                stock_data.append({
                    'Symbol': symbol,
                    'Company': info.get('longName', symbol),
                    'Price': f"${current_price:.2f}",
                    'Market Cap': f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A",
                    'Sector': info.get('sector', 'N/A')
                })
        except:
            continue
    
    if stock_data:
        df = pd.DataFrame(stock_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.error("Unable to load stock list data")

def get_sample_symbols():
    """Get a sample of stock symbols for screening"""
    return [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
        'JNJ', 'PFE', 'UNH', 'ABBV', 'BMY', 'MRK', 'CVS',
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD',
        'DIS', 'CMCSA', 'VZ', 'T', 'ORCL', 'CRM', 'ADBE',
        'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT',
        'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX'
    ]

def get_sp500_symbols():
    """Get S&P 500 symbol sample"""
    return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'META', 'UNH', 'JNJ', 'V']

def get_nasdaq100_symbols():
    """Get NASDAQ 100 symbol sample"""
    return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM']

def get_dow30_symbols():
    """Get DOW 30 symbols"""
    return ['AAPL', 'MSFT', 'BA', 'CAT', 'CVX', 'GS', 'HD', 'IBM', 'JNJ', 'JPM']

def calculate_volume_metrics(ticker_symbol):
    """Calculate volume metrics for a given stock"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get 3 months of historical data for volume calculations
        hist = ticker.history(period="3mo")
        
        if hist.empty or len(hist) < 20:
            return None
        
        current_volume = hist['Volume'].iloc[-1]
        
        # Calculate 20-day average volume
        if len(hist) >= 20:
            avg_volume_20d = hist['Volume'].tail(20).mean()
        else:
            avg_volume_20d = hist['Volume'].mean()
        
        # Calculate 3-month average volume
        avg_volume_3m = hist['Volume'].mean()
        
        # Calculate volume ratios
        volume_ratio_20d = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0
        volume_ratio_3m = current_volume / avg_volume_3m if avg_volume_3m > 0 else 0
        
        return {
            'current_volume': current_volume,
            'avg_volume_20d': avg_volume_20d,
            'avg_volume_3m': avg_volume_3m,
            'volume_ratio_20d': volume_ratio_20d,
            'volume_ratio_3m': volume_ratio_3m
        }
        
    except Exception as e:
        return None

def format_volume(volume):
    """Format volume for display"""
    if volume >= 1e9:
        return f"{volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"{volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"{volume/1e3:.2f}K"
    else:
        return f"{volume:.0f}"

def check_volume_filters(volume_metrics, min_volume, volume_comparison, volume_ratio_filter, volume_ratio_threshold):
    """Check if stock passes volume filters"""
    if not volume_metrics:
        return False
    
    # Check minimum volume threshold
    if min_volume > 0 and volume_metrics['current_volume'] < min_volume:
        return False
    
    # Check volume comparison
    if volume_comparison == "Above 20-day Average":
        if volume_metrics['volume_ratio_20d'] <= 1.0:
            return False
    elif volume_comparison == "Below 20-day Average":
        if volume_metrics['volume_ratio_20d'] >= 1.0:
            return False
    elif volume_comparison == "Above 3-month Average":
        if volume_metrics['volume_ratio_3m'] <= 1.0:
            return False
    elif volume_comparison == "Below 3-month Average":
        if volume_metrics['volume_ratio_3m'] >= 1.0:
            return False
    
    # Check volume ratio filter
    if volume_ratio_filter == "Above Ratio Threshold":
        if volume_metrics['volume_ratio_20d'] < volume_ratio_threshold:
            return False
    elif volume_ratio_filter == "Below Ratio Threshold":
        if volume_metrics['volume_ratio_20d'] > volume_ratio_threshold:
            return False
    
    return True
