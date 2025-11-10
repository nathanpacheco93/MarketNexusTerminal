import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def display_options():
    """Display options chain and derivatives pricing section"""
    
    st.subheader("📊 Options Chain & Derivatives Pricing")
    
    # Symbol input
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("Stock Symbol for Options", value="AAPL", placeholder="e.g., AAPL, TSLA, SPY")
    
    with col2:
        option_type = st.selectbox("Option Type", ["calls", "puts", "both"])
    
    with col3:
        refresh_data = st.button("🔄 Refresh Options Data")
    
    if symbol:
        try:
            # Get ticker object
            ticker = yf.Ticker(symbol.upper())
            
            # Get current stock price (try fast_info first for more recent data)
            try:
                current_price = ticker.fast_info.get('lastPrice')
                if current_price is None or current_price <= 0:
                    raise ValueError("No fast_info price available")
                price_source = "Near real-time"
            except:
                # Fallback to historical data
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    price_source = "Previous close (delayed)"
                else:
                    current_price = 100  # fallback
                    price_source = "Estimate"
            
            st.metric(f"{symbol.upper()} Current Price", f"${current_price:.2f}", help=f"Source: {price_source}")
            
            # Get options expiration dates
            try:
                expirations = ticker.options
                if expirations:
                    selected_exp = st.selectbox("Select Expiration Date", expirations)
                    
                    # Get options chain for selected expiration
                    options_chain = ticker.option_chain(selected_exp)
                    
                    if option_type in ["calls", "both"]:
                        st.subheader("📈 Call Options")
                        display_options_table(options_chain.calls, "calls", current_price, selected_exp)
                    
                    if option_type in ["puts", "both"]:
                        st.subheader("📉 Put Options")
                        display_options_table(options_chain.puts, "puts", current_price, selected_exp)
                    
                    # Options analytics section
                    st.subheader("🧮 Options Analytics & Pricing Models")
                    
                    # Black-Scholes calculator
                    display_black_scholes_calculator(current_price, selected_exp)
                    
                    # Options Greeks analysis
                    if option_type in ["calls", "both"] and not options_chain.calls.empty:
                        display_greeks_analysis(options_chain.calls, "calls", current_price, selected_exp)
                    
                    if option_type in ["puts", "both"] and not options_chain.puts.empty:
                        display_greeks_analysis(options_chain.puts, "puts", current_price, selected_exp)
                    
                    # Volatility analysis
                    display_volatility_analysis(symbol.upper(), current_price)
                    
                else:
                    st.warning(f"No options data available for {symbol.upper()}. This could be because:")
                    st.write("• The stock doesn't have listed options")
                    st.write("• The symbol is incorrect")
                    st.write("• Options data is not available through the current data provider")
                    
                    # Show Black-Scholes calculator anyway
                    st.subheader("🧮 Black-Scholes Options Pricing Calculator")
                    display_black_scholes_calculator(current_price, None)
                    
            except Exception as e:
                st.error(f"Error fetching options data: {str(e)}")
                st.info("Showing Black-Scholes calculator with manual inputs")
                display_black_scholes_calculator(current_price, None)
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")

def display_options_table(options_df, option_type, current_price, expiration):
    """Display options data in a formatted table"""
    
    if options_df.empty:
        st.warning(f"No {option_type} options available for this expiration")
        return
    
    # Calculate additional metrics
    options_df = options_df.copy()
    
    # Calculate moneyness
    if option_type == "calls":
        options_df['Moneyness'] = current_price / options_df['strike']
        options_df['Intrinsic Value'] = np.maximum(current_price - options_df['strike'], 0)
    else:
        options_df['Moneyness'] = options_df['strike'] / current_price
        options_df['Intrinsic Value'] = np.maximum(options_df['strike'] - current_price, 0)
    
    # Calculate time value
    options_df['Time Value'] = options_df['lastPrice'] - options_df['Intrinsic Value']
    
    # Format columns for display
    display_df = options_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 
                           'impliedVolatility', 'Intrinsic Value', 'Time Value', 'Moneyness']].copy()
    
    # Round numeric columns
    numeric_columns = ['lastPrice', 'bid', 'ask', 'impliedVolatility', 'Intrinsic Value', 'Time Value', 'Moneyness']
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)
    
    # Style the dataframe
    def highlight_itm(row):
        """Highlight in-the-money options"""
        if option_type == "calls" and row['strike'] < current_price:
            return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
        elif option_type == "puts" and row['strike'] > current_price:
            return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_itm, axis=1)
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Options chain visualization
    if len(options_df) > 0:
        st.subheader(f"📊 {option_type.title()} Options Visualization")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Volume by Strike', 'Open Interest by Strike',
                'Implied Volatility by Strike', 'Time Value by Strike'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=options_df['strike'], y=options_df['volume'], 
                   name='Volume', marker_color='blue'),
            row=1, col=1
        )
        
        # Open Interest
        fig.add_trace(
            go.Bar(x=options_df['strike'], y=options_df['openInterest'], 
                   name='Open Interest', marker_color='orange'),
            row=1, col=2
        )
        
        # Implied Volatility
        fig.add_trace(
            go.Scatter(x=options_df['strike'], y=options_df['impliedVolatility'], 
                      mode='lines+markers', name='IV', line=dict(color='red')),
            row=2, col=1
        )
        
        # Time Value
        fig.add_trace(
            go.Scatter(x=options_df['strike'], y=options_df['Time Value'], 
                      mode='lines+markers', name='Time Value', line=dict(color='green')),
            row=2, col=2
        )
        
        # Add vertical line for current price
        fig.add_vline(x=current_price, line_dash="dash", line_color="black", 
                     annotation_text=f"Current: ${current_price:.2f}")
        
        fig.update_layout(
            title=f"{option_type.title()} Options Analysis",
            template="plotly_dark",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_black_scholes_calculator(current_price, expiration):
    """Display Black-Scholes options pricing calculator"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option Parameters:**")
        
        # Calculate days to expiration if expiration date is provided
        if expiration:
            exp_date = datetime.strptime(expiration, "%Y-%m-%d")
            days_to_exp = (exp_date - datetime.now()).days
            default_time = max(days_to_exp / 365.0, 0.01)  # Convert to years
        else:
            default_time = 0.25  # 3 months default
        
        spot_price = st.number_input("Current Stock Price ($)", value=float(current_price), min_value=0.01)
        strike_price = st.number_input("Strike Price ($)", value=float(current_price), min_value=0.01)
        time_to_exp = st.number_input("Time to Expiration (Years)", value=default_time, min_value=0.001, max_value=10.0)
        volatility = st.number_input("Implied Volatility (%)", value=25.0, min_value=0.1, max_value=200.0) / 100
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0) / 100
        dividend_yield = st.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, max_value=20.0) / 100
    
    with col2:
        st.markdown("**Calculated Prices & Greeks:**")
        
        # Calculate Black-Scholes prices
        call_price, put_price = black_scholes_price(spot_price, strike_price, time_to_exp, 
                                                   risk_free_rate, volatility, dividend_yield)
        
        # Calculate Greeks
        greeks = calculate_greeks(spot_price, strike_price, time_to_exp, 
                                risk_free_rate, volatility, dividend_yield)
        
        # Display results
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.metric("Call Price", f"${call_price:.4f}")
            st.metric("Delta (Call)", f"{greeks['call_delta']:.4f}")
            st.metric("Gamma", f"{greeks['gamma']:.4f}")
            st.metric("Vega", f"{greeks['vega']:.4f}")
        
        with col2b:
            st.metric("Put Price", f"${put_price:.4f}")
            st.metric("Delta (Put)", f"{greeks['put_delta']:.4f}")
            st.metric("Theta (Put)", f"{greeks['put_theta']:.4f}")
            st.metric("Rho (Put)", f"{greeks['put_rho']:.4f}")
    
    # Payoff and P&L diagrams
    st.subheader("📈 Option Payoff & P&L Diagrams")
    
    # Create payoff diagram
    spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
    
    call_payoffs = np.maximum(spot_range - strike_price, 0)
    put_payoffs = np.maximum(strike_price - spot_range, 0)
    
    # P&L including premium (cost of option)
    call_pnl = call_payoffs - call_price
    put_pnl = put_payoffs - put_price
    
    # Current option values across price range
    call_values = []
    put_values = []
    
    for s in spot_range:
        c, p = black_scholes_price(s, strike_price, time_to_exp, risk_free_rate, volatility, dividend_yield)
        call_values.append(c)
        put_values.append(p)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Call Option Payoff', 'Put Option Payoff', 'Call P&L (incl. Premium)', 'Put P&L (incl. Premium)')
    )
    
    # Call option payoff
    fig.add_trace(
        go.Scatter(x=spot_range, y=call_payoffs, mode='lines', 
                  name='Expiration Payoff', line=dict(color='blue', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=spot_range, y=call_values, mode='lines', 
                  name='Current Value', line=dict(color='red')),
        row=1, col=1
    )
    
    # Put option payoff
    fig.add_trace(
        go.Scatter(x=spot_range, y=put_payoffs, mode='lines', 
                  name='Expiration Payoff', line=dict(color='blue', dash='dash'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=spot_range, y=put_values, mode='lines', 
                  name='Current Value', line=dict(color='red'), showlegend=False),
        row=1, col=2
    )
    
    # Call P&L with premium
    fig.add_trace(
        go.Scatter(x=spot_range, y=call_pnl, mode='lines', 
                  name='P&L at Expiration', line=dict(color='green')),
        row=2, col=1
    )
    
    # Put P&L with premium
    fig.add_trace(
        go.Scatter(x=spot_range, y=put_pnl, mode='lines', 
                  name='P&L at Expiration', line=dict(color='green'), showlegend=False),
        row=2, col=2
    )
    
    # Add breakeven lines
    call_breakeven = strike_price + call_price
    put_breakeven = strike_price - put_price
    
    # Add vertical lines for current price, strike, and breakeven
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_vline(x=spot_price, line_dash="dot", line_color="gray", 
                         annotation_text=f"Current: ${spot_price:.2f}", row=row, col=col)
            fig.add_vline(x=strike_price, line_dash="dot", line_color="orange", 
                         annotation_text=f"Strike: ${strike_price:.2f}", row=row, col=col)
    
    # Add breakeven lines to P&L charts
    fig.add_vline(x=call_breakeven, line_dash="dot", line_color="purple", 
                 annotation_text=f"BE: ${call_breakeven:.2f}", row=2, col=1)
    fig.add_vline(x=put_breakeven, line_dash="dot", line_color="purple", 
                 annotation_text=f"BE: ${put_breakeven:.2f}", row=2, col=2)
    
    # Add horizontal zero line for P&L charts
    fig.add_hline(y=0, line_dash="solid", line_color="white", row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="white", row=2, col=2)
    
    fig.update_layout(
        title="Option Payoff & P&L Analysis",
        template="plotly_dark",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Stock Price ($)")
    fig.update_yaxes(title_text="Option Value ($)", row=1)
    fig.update_yaxes(title_text="P&L ($)", row=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display breakeven analysis
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Call Breakeven", f"${call_breakeven:.2f}", f"{((call_breakeven/spot_price - 1)*100):+.2f}%")
    with col2:
        st.metric("Put Breakeven", f"${put_breakeven:.2f}", f"{((put_breakeven/spot_price - 1)*100):+.2f}%")

def display_greeks_analysis(options_df, option_type, current_price, expiration):
    """Display Greeks analysis for options"""
    
    if options_df.empty:
        return
    
    st.subheader(f"🔢 Greeks Analysis - {option_type.title()}")
    
    # Calculate Greeks for each option (simplified)
    exp_date = datetime.strptime(expiration, "%Y-%m-%d")
    time_to_exp = max((exp_date - datetime.now()).days / 365.0, 0.001)
    
    greeks_data = []
    
    for _, option in options_df.iterrows():
        if pd.notna(option['impliedVolatility']) and option['impliedVolatility'] > 0:
            greeks = calculate_greeks(
                current_price, option['strike'], time_to_exp,
                0.05, option['impliedVolatility'], 0.0  # Simplified assumptions
            )
            
            greeks_data.append({
                'Strike': option['strike'],
                'Delta': greeks['call_delta'] if option_type == "calls" else greeks['put_delta'],
                'Gamma': greeks['gamma'],
                'Theta': greeks['call_theta'] if option_type == "calls" else greeks['put_theta'],
                'Vega': greeks['vega'],
                'Volume': option['volume'],
                'Open Interest': option['openInterest']
            })
    
    if greeks_data:
        greeks_df = pd.DataFrame(greeks_data)
        
        # Create Greeks visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta by Strike', 'Gamma by Strike', 'Theta by Strike', 'Vega by Strike')
        )
        
        fig.add_trace(
            go.Scatter(x=greeks_df['Strike'], y=greeks_df['Delta'], 
                      mode='lines+markers', name='Delta', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=greeks_df['Strike'], y=greeks_df['Gamma'], 
                      mode='lines+markers', name='Gamma', line=dict(color='red')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=greeks_df['Strike'], y=greeks_df['Theta'], 
                      mode='lines+markers', name='Theta', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=greeks_df['Strike'], y=greeks_df['Vega'], 
                      mode='lines+markers', name='Vega', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.add_vline(x=current_price, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title=f"{option_type.title()} Greeks Analysis",
            template="plotly_dark",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display Greeks table
        st.dataframe(greeks_df.round(4), use_container_width=True)

def display_volatility_analysis(symbol, current_price):
    """Display volatility analysis"""
    
    st.subheader("📊 Volatility Analysis")
    
    try:
        # Get historical data for volatility calculation
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(period="1y")
        
        if not hist_data.empty:
            # Calculate historical volatility
            returns = hist_data['Close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate different period volatilities
            vol_30d = returns.tail(30).std() * np.sqrt(252) if len(returns) >= 30 else hist_vol
            vol_60d = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else hist_vol
            vol_90d = returns.tail(90).std() * np.sqrt(252) if len(returns) >= 90 else hist_vol
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("30-Day Volatility", f"{vol_30d*100:.2f}%")
            with col2:
                st.metric("60-Day Volatility", f"{vol_60d*100:.2f}%")
            with col3:
                st.metric("90-Day Volatility", f"{vol_90d*100:.2f}%")
            with col4:
                st.metric("1-Year Volatility", f"{hist_vol*100:.2f}%")
            
            # Volatility chart
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol * 100,
                    mode='lines',
                    name='30-Day Rolling Volatility',
                    line=dict(color='orange')
                )
            )
            
            fig.update_layout(
                title="Historical Volatility (30-Day Rolling)",
                template="plotly_dark",
                height=300,
                xaxis_title="Date",
                yaxis_title="Volatility (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Could not fetch historical data for volatility analysis")
            
    except Exception as e:
        st.error(f"Error in volatility analysis: {str(e)}")

def black_scholes_price(S, K, T, r, sigma, q=0):
    """
    Calculate Black-Scholes option prices
    
    S: Current stock price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate
    sigma: Volatility
    q: Dividend yield
    """
    try:
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        
        return call_price, put_price
    except Exception as e:
        return 0, 0

def calculate_greeks(S, K, T, r, sigma, q=0):
    """
    Calculate option Greeks
    """
    try:
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Delta
        call_delta = np.exp(-q*T) * norm.cdf(d1)
        put_delta = -np.exp(-q*T) * norm.cdf(-d1)
        
        # Gamma
        gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta (per day) - separate formulas for calls and puts
        call_theta = (-S*np.exp(-q*T)*norm.pdf(d1)*sigma/(2*np.sqrt(T)) 
                     - r*K*np.exp(-r*T)*norm.cdf(d2) 
                     + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        
        put_theta = (-S*np.exp(-q*T)*norm.pdf(d1)*sigma/(2*np.sqrt(T)) 
                    + r*K*np.exp(-r*T)*norm.cdf(-d2) 
                    - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        
        # Vega (per 1% change in volatility)
        vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho (per 1% change in interest rate)
        call_rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        put_rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
        
        return {
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'call_theta': call_theta,
            'put_theta': put_theta,
            'vega': vega,
            'call_rho': call_rho,
            'put_rho': put_rho
        }
    except Exception as e:
        return {
            'call_delta': 0, 'put_delta': 0, 'gamma': 0, 'call_theta': 0, 'put_theta': 0,
            'vega': 0, 'call_rho': 0, 'put_rho': 0
        }