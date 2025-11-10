import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from modules.user_auth import user_auth

def display_charts():
    """Display interactive charts section"""
    
    # Symbol input
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, GOOGL, TSLA")
    
    with col2:
        default_period = user_auth.get_user_preference('charts', 'default_period', '1y')
        period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                            index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"].index(default_period))
        user_auth.auto_save_preference('charts', 'default_period', period)
    
    with col3:
        default_chart_type = user_auth.get_user_preference('charts', 'default_type', 'Candlestick')
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"],
                                index=["Candlestick", "Line", "OHLC"].index(default_chart_type))
        user_auth.auto_save_preference('charts', 'default_type', chart_type)
    
    if symbol:
        try:
            # Get stock data
            ticker = yf.Ticker(symbol.upper())
            data = ticker.history(period=period)
            info = ticker.info
            
            if not data.empty:
                # Display stock info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = data['Close'].iloc[-1]
                    prev_close = info.get('previousClose', data['Close'].iloc[-2] if len(data) > 1 else current_price)
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100
                    
                    st.metric(
                        label=f"{symbol.upper()} - {info.get('longName', symbol.upper())}",
                        value=f"${current_price:.2f}",
                        delta=f"{change:.2f} ({change_percent:.2f}%)"
                    )
                
                with col2:
                    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                
                with col3:
                    st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A")
                
                with col4:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A")
                
                # Indicator options - organized into logical groups
                st.subheader("📊 Chart Options")
                
                # Create three columns for indicator groups
                trend_col, momentum_col, analysis_col = st.columns(3)
                
                with trend_col:
                    st.markdown("**📈 Trend Indicators**")
                    show_ma = st.checkbox("Moving Averages (20, 50)", 
                                        value=user_auth.get_user_preference('indicators', 'default_ma', True))
                    user_auth.auto_save_preference('indicators', 'default_ma', show_ma)
                    
                    show_ema13 = st.checkbox("13 EMA", 
                                           value=user_auth.get_user_preference('indicators', 'default_ema13', False))
                    user_auth.auto_save_preference('indicators', 'default_ema13', show_ema13)
                    
                    show_bb = st.checkbox("Bollinger Bands", 
                                        value=user_auth.get_user_preference('indicators', 'default_bb', False))
                    user_auth.auto_save_preference('indicators', 'default_bb', show_bb)
                
                with momentum_col:
                    st.markdown("**⚡ Momentum Indicators**")
                    show_stoch = st.checkbox("Stochastic Oscillator", 
                                           value=user_auth.get_user_preference('indicators', 'default_stoch', False))
                    user_auth.auto_save_preference('indicators', 'default_stoch', show_stoch)
                    
                    show_will_r = st.checkbox("Williams %R", 
                                            value=user_auth.get_user_preference('indicators', 'default_will_r', False))
                    user_auth.auto_save_preference('indicators', 'default_will_r', show_will_r)
                
                with analysis_col:
                    st.markdown("**🔍 Analysis Tools**")
                    show_fib = st.checkbox("Fibonacci Retracement", 
                                         value=user_auth.get_user_preference('indicators', 'default_fib', False))
                    user_auth.auto_save_preference('indicators', 'default_fib', show_fib)
                
                # Create main chart with multiple subplots
                rows = 2
                subplot_count = 0
                if show_stoch:
                    subplot_count += 1
                if show_will_r:
                    subplot_count += 1
                rows += subplot_count
                
                subplot_titles = [f'{symbol.upper()} Price Chart', 'Volume']
                row_heights = [0.7, 0.3]
                
                # Adjust row heights and titles based on indicators
                if subplot_count == 1:
                    row_heights = [0.6, 0.2, 0.2]
                elif subplot_count == 2:
                    row_heights = [0.5, 0.2, 0.15, 0.15]
                
                if show_stoch:
                    subplot_titles.append('Stochastic Oscillator')
                if show_will_r:
                    subplot_titles.append('Williams %R')
                
                fig = make_subplots(
                    rows=rows, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=subplot_titles,
                    row_heights=row_heights
                )
                
                # Price chart
                if chart_type == "Candlestick":
                    fig.add_trace(
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="OHLC"
                        ),
                        row=1, col=1
                    )
                elif chart_type == "Line":
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='#FF6B35', width=2)
                        ),
                        row=1, col=1
                    )
                else:  # OHLC
                    fig.add_trace(
                        go.Ohlc(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="OHLC"
                        ),
                        row=1, col=1
                    )
                
                # Add moving averages
                if show_ma and len(data) >= 20:
                    data['MA20'] = data['Close'].rolling(window=20).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['MA20'],
                            mode='lines',
                            name='MA 20',
                            line=dict(color='blue', width=1)
                        ),
                        row=1, col=1
                    )
                
                if show_ma and len(data) >= 50:
                    data['MA50'] = data['Close'].rolling(window=50).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['MA50'],
                            mode='lines',
                            name='MA 50',
                            line=dict(color='red', width=1)
                        ),
                        row=1, col=1
                    )
                
                # Add 13 EMA
                if show_ema13 and len(data) >= 13:
                    data['EMA13'] = calculate_ema_13(data['Close'])
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['EMA13'],
                            mode='lines',
                            name='EMA 13',
                            line=dict(color='purple', width=1)
                        ),
                        row=1, col=1
                    )
                
                # Add Bollinger Bands overlay
                if show_bb and len(data) >= 20:
                    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands_series(data['Close'], 20, 2)
                    
                    # Upper band
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=bb_upper,
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='rgba(173,216,230,0.5)', width=1),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Lower band with fill
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=bb_lower,
                            mode='lines',
                            name='Bollinger Bands',
                            line=dict(color='rgba(173,216,230,0.5)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(173,216,230,0.1)'
                        ),
                        row=1, col=1
                    )
                    
                    # Middle band (SMA)
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=bb_middle,
                            mode='lines',
                            name='BB Middle',
                            line=dict(color='orange', width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                
                # Add Fibonacci retracement levels overlay
                if show_fib and len(data) >= 20:
                    lookback = min(50, len(data))
                    recent_data = data.tail(lookback)
                    high_price = recent_data['High'].max()
                    low_price = recent_data['Low'].min()
                    fib_levels = calculate_fibonacci_levels(high_price, low_price)
                    
                    # Add horizontal lines for key Fibonacci levels
                    for level_name, price in fib_levels.items():
                        if any(fib in level_name for fib in ['23.6%', '38.2%', '50%', '61.8%']):
                            fig.add_hline(
                                y=price,
                                line=dict(color='gold', width=1, dash='dot'),
                                annotation_text=f"{level_name}: ${price:.2f}",
                                annotation_position="right",
                                row=1
                            )
                
                # Volume chart
                colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                         for i in range(len(data))]
                
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
                
                # Stochastic Oscillator chart
                current_row = 3
                if show_stoch and len(data) >= 14:
                    stoch_k_series, stoch_d_series = calculate_stochastic_series(data, 14, 3)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=stoch_k_series,
                            mode='lines',
                            name='%K',
                            line=dict(color='blue', width=2)
                        ),
                        row=current_row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=stoch_d_series,
                            mode='lines',
                            name='%D',
                            line=dict(color='red', width=2)
                        ),
                        row=current_row, col=1
                    )
                    
                    # Add overbought/oversold lines
                    fig.add_hline(y=80, line=dict(color='red', dash='dash'), row=current_row, col=1)
                    fig.add_hline(y=20, line=dict(color='green', dash='dash'), row=current_row, col=1)
                    current_row += 1
                
                # Williams %R chart
                if show_will_r and len(data) >= 14:
                    will_r_series = calculate_williams_r_series(data, 14)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=will_r_series,
                            mode='lines',
                            name='Williams %R',
                            line=dict(color='orange', width=2)
                        ),
                        row=current_row, col=1
                    )
                    
                    # Add overbought/oversold lines for Williams %R
                    fig.add_hline(y=-20, line=dict(color='red', dash='dash'), row=current_row, col=1)
                    fig.add_hline(y=-80, line=dict(color='green', dash='dash'), row=current_row, col=1)
                
                # Update layout with dynamic height
                height = 700 + (subplot_count * 150)  # Base height + 150px per indicator subplot
                
                fig.update_layout(
                    title=f"{symbol.upper()} - {period.upper()} Chart",
                    template="plotly_dark",
                    height=height,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                # Update y-axis labels
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                # Set y-axis for indicator subplots
                axis_row = 3
                if show_stoch:
                    fig.update_yaxes(title_text="Stochastic %", row=axis_row, col=1, range=[0, 100])
                    axis_row += 1
                if show_will_r:
                    fig.update_yaxes(title_text="Williams %R", row=axis_row, col=1, range=[-100, 0])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # MACD Chart (separate chart)
                if len(data) >= 26:
                    st.subheader("📈 MACD Analysis")
                    show_macd = st.checkbox("Show MACD Chart", value=True)
                    
                    if show_macd:
                        macd_line, macd_signal, macd_histogram = calculate_macd(data['Close'])
                        
                        if len(macd_line) > 0:
                            # Create MACD subplot
                            macd_fig = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=('MACD Line & Signal', 'MACD Histogram'),
                                row_heights=[0.7, 0.3]
                            )
                            
                            # MACD Line and Signal - use proper index alignment
                            macd_fig.add_trace(
                                go.Scatter(
                                    x=macd_line.index,
                                    y=macd_line.values,
                                    mode='lines',
                                    name='MACD Line',
                                    line=dict(color='blue', width=2)
                                ),
                                row=1, col=1
                            )
                            
                            macd_fig.add_trace(
                                go.Scatter(
                                    x=macd_signal.index,
                                    y=macd_signal.values,
                                    mode='lines',
                                    name='Signal Line',
                                    line=dict(color='red', width=2)
                                ),
                                row=1, col=1
                            )
                            
                            # MACD Histogram
                            colors = ['red' if x < 0 else 'green' for x in macd_histogram.values]
                            macd_fig.add_trace(
                                go.Bar(
                                    x=macd_histogram.index,
                                    y=macd_histogram.values,
                                    name='MACD Histogram',
                                    marker_color=colors,
                                    opacity=0.7
                                ),
                                row=2, col=1
                            )
                            
                            macd_fig.update_layout(
                                title="MACD Analysis",
                                template="plotly_dark",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(macd_fig, use_container_width=True)
                            
                            # MACD interpretation
                            current_macd = macd_line.iloc[-1] if len(macd_line) > 0 else 0
                            current_signal = macd_signal.iloc[-1] if len(macd_signal) > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MACD Value", f"{current_macd:.4f}")
                            with col2:
                                st.metric("Signal Value", f"{current_signal:.4f}")
                            with col3:
                                if current_macd > current_signal:
                                    st.success("📈 Bullish Signal")
                                else:
                                    st.warning("📉 Bearish Signal")
                
                # Technical indicators section
                st.subheader("🔍 Technical Indicators")
                
                # Display indicators in two rows to accommodate Williams %R
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # RSI calculation
                    if len(data) >= 14:
                        rsi = calculate_rsi_wilder(data['Close'], 14)
                        st.metric("RSI (14)", f"{rsi:.2f}")
                        
                        # RSI interpretation
                        if rsi > 70:
                            st.warning("⚠️ Overbought")
                        elif rsi < 30:
                            st.success("💡 Oversold")
                        else:
                            st.info("📊 Neutral")
                
                with col2:
                    # Bollinger Bands
                    if len(data) >= 20:
                        bb_upper_val, bb_middle_val, bb_lower_val = calculate_bollinger_bands_values(data['Close'], 20, 2)
                        current_price = data['Close'].iloc[-1]
                        
                        if bb_upper_val > bb_lower_val:  # Avoid division by zero
                            bb_position = ((current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)) * 100
                            st.metric("BB Position", f"{bb_position:.1f}%")
                            
                            if bb_position > 80:
                                st.warning("⚠️ Near upper")
                            elif bb_position < 20:
                                st.success("💡 Near lower")
                            else:
                                st.info("📊 Middle range")
                        else:
                            st.info("BB: Insufficient data")
                
                with col3:
                    # Stochastic Oscillator
                    if len(data) >= 14:
                        stoch_k, stoch_d = calculate_stochastic(data, 14)
                        st.metric("Stochastic %K", f"{stoch_k:.2f}")
                        
                        if stoch_k > 80:
                            st.warning("⚠️ Overbought")
                        elif stoch_k < 20:
                            st.success("💡 Oversold")
                        else:
                            st.info("📊 Neutral")
                
                with col4:
                    # Williams %R
                    if len(data) >= 14:
                        will_r = calculate_williams_r(data, 14)
                        st.metric("Williams %R", f"{will_r:.2f}")
                        
                        # Williams %R interpretation (overbought >-20, oversold <-80)
                        if will_r > -20:
                            st.warning("⚠️ Overbought")
                        elif will_r < -80:
                            st.success("💡 Oversold")
                        else:
                            st.info("📊 Neutral")
                
                # Performance metrics
                st.subheader("📈 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                if len(data) >= 5:
                    with col1:
                        week_return = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100
                        st.metric("5-Day Return", f"{week_return:.2f}%")
                
                if len(data) >= 22:
                    with col2:
                        month_return = ((data['Close'].iloc[-1] / data['Close'].iloc[-22]) - 1) * 100
                        st.metric("1-Month Return", f"{month_return:.2f}%")
                
                if len(data) >= 66:
                    with col3:
                        quarter_return = ((data['Close'].iloc[-1] / data['Close'].iloc[-66]) - 1) * 100
                        st.metric("3-Month Return", f"{quarter_return:.2f}%")
                
                if len(data) >= 252:
                    with col4:
                        year_return = ((data['Close'].iloc[-1] / data['Close'].iloc[-252]) - 1) * 100
                        st.metric("1-Year Return", f"{year_return:.2f}%")
                
            else:
                st.error("No data available for this symbol. Please check the symbol and try again.")
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")

def calculate_rsi_wilder(prices, window=14):
    """Calculate RSI using Wilder's smoothing method"""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing (alpha = 1/window)
        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50
    except Exception as e:
        return 50

def calculate_bollinger_bands_series(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands - full series"""
    try:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.fillna(method='bfill'), sma.fillna(method='bfill'), lower_band.fillna(method='bfill')
    except Exception as e:
        empty_series = pd.Series([0] * len(prices), index=prices.index)
        return empty_series, empty_series, empty_series

def calculate_bollinger_bands_values(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands - single values"""
    try:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        upper_val = upper_band.iloc[-1] if not upper_band.empty and not pd.isna(upper_band.iloc[-1]) else 0
        middle_val = sma.iloc[-1] if not sma.empty and not pd.isna(sma.iloc[-1]) else 0
        lower_val = lower_band.iloc[-1] if not lower_band.empty and not pd.isna(lower_band.iloc[-1]) else 0
        
        return upper_val, middle_val, lower_val
    except Exception as e:
        return 0, 0, 0

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.dropna(), signal_line.dropna(), histogram.dropna()
    except Exception as e:
        # Return empty series if calculation fails
        return pd.Series(), pd.Series(), pd.Series()

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    levels = {
        '0% (High)': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100% (Low)': low
    }
    return levels

def calculate_stochastic(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator - single values"""
    try:
        low_k = data['Low'].rolling(window=k_period).min()
        high_k = data['High'].rolling(window=k_period).max()
        
        # Handle division by zero
        denominator = high_k - low_k
        k_percent = 100 * ((data['Close'] - low_k) / denominator.replace(0, 1))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        # Return last valid values
        k_val = k_percent.iloc[-1] if not k_percent.empty and not pd.isna(k_percent.iloc[-1]) else 0
        d_val = d_percent.iloc[-1] if not d_percent.empty and not pd.isna(d_percent.iloc[-1]) else 0
        
        return k_val, d_val
    except Exception as e:
        return 0, 0

def calculate_stochastic_series(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator - full series"""
    try:
        low_k = data['Low'].rolling(window=k_period).min()
        high_k = data['High'].rolling(window=k_period).max()
        
        # Handle division by zero
        denominator = high_k - low_k
        k_percent = 100 * ((data['Close'] - low_k) / denominator.replace(0, 1))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent.fillna(0), d_percent.fillna(0)
    except Exception as e:
        empty_series = pd.Series([0] * len(data), index=data.index)
        return empty_series, empty_series

# Legacy function for compatibility
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index - legacy function"""
    return calculate_rsi_wilder(prices, window)

def calculate_bollinger_bands(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands - legacy function for compatibility"""
    upper_val, middle_val, lower_val = calculate_bollinger_bands_values(prices, window, std_dev)
    return upper_val, lower_val

def calculate_williams_r(data, period=14):
    """Calculate Williams %R - single value"""
    try:
        highest_high = data['High'].rolling(window=period).max()
        lowest_low = data['Low'].rolling(window=period).min()
        
        # Williams %R formula: %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
        denominator = highest_high - lowest_low
        williams_r = -100 * ((highest_high - data['Close']) / denominator.replace(0, 1))
        
        # Return last valid value
        return williams_r.iloc[-1] if not williams_r.empty and not pd.isna(williams_r.iloc[-1]) else -50
    except Exception as e:
        return -50

def calculate_williams_r_series(data, period=14):
    """Calculate Williams %R - full series"""
    try:
        highest_high = data['High'].rolling(window=period).max()
        lowest_low = data['Low'].rolling(window=period).min()
        
        # Williams %R formula: %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
        denominator = highest_high - lowest_low
        williams_r = -100 * ((highest_high - data['Close']) / denominator.replace(0, 1))
        
        return williams_r.fillna(-50)
    except Exception as e:
        return pd.Series([-50] * len(data), index=data.index)

def calculate_ema_13(prices):
    """Calculate 13-period Exponential Moving Average"""
    try:
        ema_13 = prices.ewm(span=13, adjust=False).mean()
        return ema_13.fillna(method='bfill')
    except Exception as e:
        return pd.Series([0] * len(prices), index=prices.index)