import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

def display_bonds():
    """Display bond market data and fixed income analytics"""
    
    st.subheader("🏛️ Bond Market & Fixed Income Analytics")
    
    # Treasury yield curve section
    st.markdown("### 📈 Treasury Yield Curve")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Yield curve controls
        curve_date = st.date_input("Curve Date", value=datetime.now().date())
        historical_comparison = st.checkbox("Show Historical Comparison")
        
        if historical_comparison:
            comparison_periods = st.multiselect(
                "Compare with",
                ["1 Month Ago", "3 Months Ago", "6 Months Ago", "1 Year Ago"],
                default=["3 Months Ago", "1 Year Ago"]
            )
        
        refresh_data = st.button("🔄 Refresh Treasury Data")
    
    with col1:
        # Fetch and display yield curve
        yield_data = fetch_treasury_yields()
        
        if yield_data is not None:
            display_yield_curve(yield_data, historical_comparison, 
                               comparison_periods if historical_comparison else [])
        else:
            st.error("Unable to fetch treasury yield data")
    
    # Bond analytics section
    st.markdown("### 🔢 Bond Calculator & Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Bond Parameters:**")
        face_value = st.number_input("Face Value ($)", value=1000, min_value=100, step=100)
        coupon_rate = st.number_input("Coupon Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.1) / 100
        years_to_maturity = st.number_input("Years to Maturity", value=10, min_value=0.1, max_value=50.0, step=0.1)
        payment_frequency = st.selectbox("Payment Frequency", [1, 2, 4, 12], index=1, 
                                       format_func=lambda x: {1: "Annual", 2: "Semi-Annual", 4: "Quarterly", 12: "Monthly"}[x])
    
    with col2:
        st.markdown("**Market Data:**")
        yield_to_maturity = st.number_input("Yield to Maturity (%)", value=4.5, min_value=0.0, max_value=20.0, step=0.1) / 100
        current_price = st.number_input("Current Price ($)", value=1050.0, min_value=100.0, step=1.0)
        
        # Calculate bond metrics
        bond_metrics = calculate_bond_metrics(face_value, coupon_rate, years_to_maturity, 
                                            payment_frequency, yield_to_maturity, current_price)
    
    with col3:
        st.markdown("**Calculated Metrics:**")
        if bond_metrics:
            st.metric("Bond Price", f"${bond_metrics['price']:.2f}")
            st.metric("Duration", f"{bond_metrics['duration']:.2f} years")
            st.metric("Modified Duration", f"{bond_metrics['modified_duration']:.2f}")
            st.metric("Convexity", f"{bond_metrics['convexity']:.2f}")
            st.metric("DV01", f"${bond_metrics['dv01']:.2f}")
    
    # Bond price sensitivity analysis
    if bond_metrics:
        st.markdown("### 📊 Price Sensitivity Analysis")
        display_bond_sensitivity_analysis(face_value, coupon_rate, years_to_maturity, 
                                         payment_frequency, yield_to_maturity)
    
    # Corporate bond screener
    st.markdown("### 🏢 Corporate Bond Analysis")
    display_corporate_bonds()
    
    # Fixed income portfolio analysis
    st.markdown("### 💼 Fixed Income Portfolio Analytics")
    display_portfolio_analytics()

def fetch_treasury_yields():
    """Fetch current treasury yield data"""
    
    try:
        # Treasury yield symbols for different maturities
        treasury_symbols = {
            '1M': '^IRX',      # 13-week treasury
            '3M': '^IRX',      # 3-month treasury
            '6M': '^IRX',      # 6-month treasury  
            '1Y': '^TNX',      # 10-year note (we'll use for 1Y approximation)
            '2Y': '^TNX',      # 2-year note
            '5Y': '^FVX',      # 5-year note
            '10Y': '^TNX',     # 10-year note
            '30Y': '^TYX'      # 30-year bond
        }
        
        # Alternative: Create synthetic yield curve data for demonstration
        # In production, you'd want to use FRED API or Bloomberg API
        current_yields = {
            '1M': 5.25,
            '3M': 5.30,
            '6M': 5.15,
            '1Y': 4.85,
            '2Y': 4.75,
            '5Y': 4.65,
            '10Y': 4.70,
            '30Y': 4.85
        }
        
        # Convert to DataFrame
        maturities = [1/12, 3/12, 6/12, 1, 2, 5, 10, 30]  # In years
        yields = list(current_yields.values())
        
        yield_data = pd.DataFrame({
            'Maturity': maturities,
            'Yield': yields,
            'Maturity_Label': list(current_yields.keys())
        })
        
        # Add some historical data for comparison (synthetic for demo)
        yield_data['Yield_3M_Ago'] = [y + np.random.normal(0, 0.2) for y in yields]
        yield_data['Yield_1Y_Ago'] = [y + np.random.normal(0, 0.5) for y in yields]
        
        return yield_data
        
    except Exception as e:
        st.error(f"Error fetching treasury data: {str(e)}")
        return None

def display_yield_curve(yield_data, show_historical, comparison_periods):
    """Display treasury yield curve"""
    
    fig = go.Figure()
    
    # Current yield curve
    fig.add_trace(
        go.Scatter(
            x=yield_data['Maturity'],
            y=yield_data['Yield'],
            mode='lines+markers',
            name='Current',
            line=dict(color='#00ff41', width=3),
            marker=dict(size=8)
        )
    )
    
    # Historical comparisons
    if show_historical:
        colors = ['#ff6b6b', '#ffa500', '#9370db']
        
        for i, period in enumerate(comparison_periods):
            if period == "3 Months Ago" and 'Yield_3M_Ago' in yield_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=yield_data['Maturity'],
                        y=yield_data['Yield_3M_Ago'],
                        mode='lines+markers',
                        name='3M Ago',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                        marker=dict(size=6)
                    )
                )
            elif period == "1 Year Ago" and 'Yield_1Y_Ago' in yield_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=yield_data['Maturity'],
                        y=yield_data['Yield_1Y_Ago'],
                        mode='lines+markers',
                        name='1Y Ago',
                        line=dict(color=colors[(i+1) % len(colors)], width=2, dash='dot'),
                        marker=dict(size=6)
                    )
                )
    
    # Add maturity labels
    for _, row in yield_data.iterrows():
        fig.add_annotation(
            x=row['Maturity'],
            y=row['Yield'],
            text=row['Maturity_Label'],
            showarrow=False,
            yshift=15,
            font=dict(size=10, color='white')
        )
    
    fig.update_layout(
        title="U.S. Treasury Yield Curve",
        template="plotly_dark",
        height=400,
        xaxis_title="Maturity (Years)",
        yaxis_title="Yield (%)",
        xaxis=dict(type='log', tickmode='array', 
                  tickvals=[1/12, 3/12, 6/12, 1, 2, 5, 10, 30],
                  ticktext=['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Yield curve analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Curve steepness (10Y - 2Y spread)
        steepness = yield_data[yield_data['Maturity_Label'] == '10Y']['Yield'].iloc[0] - \
                   yield_data[yield_data['Maturity_Label'] == '2Y']['Yield'].iloc[0]
        st.metric("2Y-10Y Spread", f"{steepness:.2f} bps")
    
    with col2:
        # Short end (3M yield)
        short_end = yield_data[yield_data['Maturity_Label'] == '3M']['Yield'].iloc[0]
        st.metric("3M Treasury", f"{short_end:.2f}%")
    
    with col3:
        # Long end (30Y yield)
        long_end = yield_data[yield_data['Maturity_Label'] == '30Y']['Yield'].iloc[0]
        st.metric("30Y Treasury", f"{long_end:.2f}%")

def calculate_bond_metrics(face_value, coupon_rate, years_to_maturity, payment_frequency, ytm, current_price):
    """Calculate comprehensive bond metrics"""
    
    try:
        # Number of payments
        n_payments = int(years_to_maturity * payment_frequency)
        
        # Periodic rates
        periodic_coupon = coupon_rate / payment_frequency
        periodic_ytm = ytm / payment_frequency
        
        # Cash flows
        coupon_payment = face_value * periodic_coupon
        
        # Bond price calculation
        if periodic_ytm == 0:
            bond_price = face_value + coupon_payment * n_payments
        else:
            pv_coupons = coupon_payment * (1 - (1 + periodic_ytm)**(-n_payments)) / periodic_ytm
            pv_face_value = face_value / (1 + periodic_ytm)**n_payments
            bond_price = pv_coupons + pv_face_value
        
        # Duration calculation (Macaulay Duration)
        duration = 0
        for t in range(1, n_payments + 1):
            if t < n_payments:
                cash_flow = coupon_payment
            else:
                cash_flow = coupon_payment + face_value
            
            pv_cash_flow = cash_flow / (1 + periodic_ytm)**t
            weight = pv_cash_flow / bond_price
            duration += weight * (t / payment_frequency)
        
        # Modified Duration
        modified_duration = duration / (1 + periodic_ytm)
        
        # Convexity calculation
        convexity = 0
        for t in range(1, n_payments + 1):
            if t < n_payments:
                cash_flow = coupon_payment
            else:
                cash_flow = coupon_payment + face_value
            
            pv_cash_flow = cash_flow / (1 + periodic_ytm)**t
            convexity += (pv_cash_flow / bond_price) * (t * (t + 1)) / ((1 + periodic_ytm)**2 * payment_frequency**2)
        
        # DV01 (Dollar Value of 01)
        dv01 = modified_duration * bond_price * 0.0001
        
        return {
            'price': bond_price,
            'duration': duration,
            'modified_duration': modified_duration,
            'convexity': convexity,
            'dv01': dv01
        }
        
    except Exception as e:
        st.error(f"Error calculating bond metrics: {str(e)}")
        return None

def display_bond_sensitivity_analysis(face_value, coupon_rate, years_to_maturity, payment_frequency, base_ytm):
    """Display bond price sensitivity to yield changes"""
    
    # Yield range for sensitivity analysis
    yield_range = np.arange(max(0.001, base_ytm - 0.03), base_ytm + 0.03, 0.001)
    
    prices = []
    durations = []
    convexities = []
    
    for ytm in yield_range:
        metrics = calculate_bond_metrics(face_value, coupon_rate, years_to_maturity, 
                                       payment_frequency, ytm, 0)
        if metrics:
            prices.append(metrics['price'])
            durations.append(metrics['duration'])
            convexities.append(metrics['convexity'])
        else:
            prices.append(None)
            durations.append(None)
            convexities.append(None)
    
    # Create sensitivity charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price vs Yield', 'Duration vs Yield', 'Convexity vs Yield', 'Price Change Approximation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price vs Yield
    fig.add_trace(
        go.Scatter(
            x=yield_range * 100,
            y=prices,
            mode='lines',
            name='Bond Price',
            line=dict(color='#00ff41', width=2)
        ),
        row=1, col=1
    )
    
    # Mark current yield
    current_price = calculate_bond_metrics(face_value, coupon_rate, years_to_maturity, 
                                         payment_frequency, base_ytm, 0)['price']
    fig.add_trace(
        go.Scatter(
            x=[base_ytm * 100],
            y=[current_price],
            mode='markers',
            name='Current',
            marker=dict(color='red', size=10, symbol='star')
        ),
        row=1, col=1
    )
    
    # Duration vs Yield
    fig.add_trace(
        go.Scatter(
            x=yield_range * 100,
            y=durations,
            mode='lines',
            name='Duration',
            line=dict(color='orange', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Convexity vs Yield
    fig.add_trace(
        go.Scatter(
            x=yield_range * 100,
            y=convexities,
            mode='lines',
            name='Convexity',
            line=dict(color='purple', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Price change approximation (Duration + Convexity)
    base_metrics = calculate_bond_metrics(face_value, coupon_rate, years_to_maturity, 
                                        payment_frequency, base_ytm, 0)
    
    yield_changes = (yield_range - base_ytm) * 100  # in basis points
    
    # Duration approximation
    duration_approx = [-base_metrics['modified_duration'] * dy / 100 * current_price for dy in yield_changes]
    
    # Duration + Convexity approximation
    duration_convexity_approx = [
        -base_metrics['modified_duration'] * dy / 100 * current_price + 
        0.5 * base_metrics['convexity'] * (dy / 100)**2 * current_price
        for dy in yield_changes
    ]
    
    # Actual price changes
    actual_changes = [p - current_price for p in prices]
    
    fig.add_trace(
        go.Scatter(
            x=yield_changes,
            y=actual_changes,
            mode='lines',
            name='Actual',
            line=dict(color='white', width=2)
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=yield_changes,
            y=duration_approx,
            mode='lines',
            name='Duration Only',
            line=dict(color='orange', width=2, dash='dash')
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=yield_changes,
            y=duration_convexity_approx,
            mode='lines',
            name='Duration + Convexity',
            line=dict(color='green', width=2, dash='dot')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Bond Sensitivity Analysis",
        template="plotly_dark",
        height=600,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Yield (%)", row=1, col=1)
    fig.update_xaxes(title_text="Yield (%)", row=1, col=2)
    fig.update_xaxes(title_text="Yield (%)", row=2, col=1)
    fig.update_xaxes(title_text="Yield Change (bps)", row=2, col=2)
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Duration", row=1, col=2)
    fig.update_yaxes(title_text="Convexity", row=2, col=1)
    fig.update_yaxes(title_text="Price Change ($)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def display_corporate_bonds():
    """Display corporate bond analysis"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Corporate Bond Screener:**")
        
        # Bond screening criteria
        min_rating = st.selectbox("Minimum Rating", ["AAA", "AA", "A", "BBB", "BB", "B"], index=2)
        max_maturity = st.slider("Maximum Maturity (Years)", 1, 30, 10)
        min_yield = st.number_input("Minimum Yield (%)", value=0.0, step=0.1)
        max_yield = st.number_input("Maximum Yield (%)", value=10.0, step=0.1)
        
        sectors = st.multiselect(
            "Sectors",
            ["Financial", "Technology", "Healthcare", "Energy", "Utilities", "Consumer", "Industrial"],
            default=["Financial", "Technology"]
        )
    
    with col2:
        st.markdown("**Sample Corporate Bonds:**")
        
        # Create sample corporate bond data
        sample_bonds = create_sample_corporate_bonds()
        
        if sample_bonds is not None:
            # Filter based on criteria
            filtered_bonds = sample_bonds[
                (sample_bonds['Maturity_Years'] <= max_maturity) &
                (sample_bonds['Yield'] >= min_yield) &
                (sample_bonds['Yield'] <= max_yield)
            ]
            
            if not filtered_bonds.empty:
                st.dataframe(filtered_bonds, use_container_width=True)
                
                # Credit spread analysis
                if st.button("📊 Analyze Credit Spreads"):
                    display_credit_spread_analysis(filtered_bonds)
            else:
                st.info("No bonds match the specified criteria")

def create_sample_corporate_bonds():
    """Create sample corporate bond data for demonstration"""
    
    try:
        bonds_data = {
            'Issuer': ['Apple Inc', 'Microsoft Corp', 'JPMorgan Chase', 'Amazon.com Inc', 'Google Inc'],
            'Coupon': [3.25, 2.88, 4.45, 3.15, 2.65],
            'Maturity': ['2025-02-23', '2024-11-03', '2026-01-23', '2027-05-12', '2025-08-15'],
            'Yield': [4.12, 3.95, 4.78, 4.32, 3.88],
            'Rating': ['AA+', 'AAA', 'A-', 'AA', 'AA+'],
            'Sector': ['Technology', 'Technology', 'Financial', 'Consumer', 'Technology'],
            'Spread': [85, 68, 145, 98, 75]  # basis points over treasury
        }
        
        df = pd.DataFrame(bonds_data)
        
        # Calculate years to maturity
        df['Maturity'] = pd.to_datetime(df['Maturity'])
        df['Maturity_Years'] = (df['Maturity'] - datetime.now()).dt.days / 365.25
        
        # Format for display
        df['Maturity_Display'] = df['Maturity'].dt.strftime('%Y-%m-%d')
        df['Yield_Display'] = df['Yield'].apply(lambda x: f"{x:.2f}%")
        df['Spread_Display'] = df['Spread'].apply(lambda x: f"{x} bps")
        
        display_df = df[['Issuer', 'Coupon', 'Maturity_Display', 'Yield_Display', 
                        'Rating', 'Sector', 'Spread_Display']].copy()
        display_df.columns = ['Issuer', 'Coupon %', 'Maturity', 'Yield', 'Rating', 'Sector', 'Spread']
        
        return df
        
    except Exception as e:
        st.error(f"Error creating bond data: {str(e)}")
        return None

def display_credit_spread_analysis(bonds_data):
    """Display credit spread analysis"""
    
    if bonds_data.empty:
        return
    
    st.subheader("💰 Credit Spread Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spread by rating
        rating_spreads = bonds_data.groupby('Rating')['Spread'].mean().sort_values(ascending=True)
        
        fig_rating = px.bar(
            x=rating_spreads.index,
            y=rating_spreads.values,
            title="Average Credit Spread by Rating",
            labels={'x': 'Rating', 'y': 'Spread (bps)'}
        )
        
        fig_rating.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        # Spread by sector
        sector_spreads = bonds_data.groupby('Sector')['Spread'].mean().sort_values(ascending=True)
        
        fig_sector = px.bar(
            x=sector_spreads.index,
            y=sector_spreads.values,
            title="Average Credit Spread by Sector",
            labels={'x': 'Sector', 'y': 'Spread (bps)'}
        )
        
        fig_sector.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_sector, use_container_width=True)
    
    # Yield vs Maturity scatter
    fig_scatter = px.scatter(
        bonds_data,
        x='Maturity_Years',
        y='Yield',
        color='Rating',
        size='Spread',
        hover_data=['Issuer', 'Sector'],
        title="Corporate Bond Yield vs Maturity"
    )
    
    fig_scatter.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Years to Maturity",
        yaxis_title="Yield (%)"
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

def display_portfolio_analytics():
    """Display fixed income portfolio analytics"""
    
    st.markdown("**Fixed Income Portfolio Builder:**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio input
        portfolio_input = st.text_area(
            "Enter Bond Portfolio (CUSIP/Name:Allocation, one per line)",
            value="US Treasury 2Y:0.3\nApple 2025:0.2\nMicrosoft 2024:0.2\nJPMorgan 2026:0.3",
            help="Format: Bond_Name:Weight"
        )
        
        portfolio = parse_bond_portfolio(portfolio_input)
        
        if portfolio:
            st.success(f"Portfolio loaded: {len(portfolio)} bonds")
            
            # Calculate portfolio duration and yield
            portfolio_metrics = calculate_portfolio_duration_yield(portfolio)
            
            if portfolio_metrics:
                col1a, col1b, col1c = st.columns(3)
                
                with col1a:
                    st.metric("Portfolio Duration", f"{portfolio_metrics['duration']:.2f} years")
                
                with col1b:
                    st.metric("Portfolio Yield", f"{portfolio_metrics['yield']:.2f}%")
                
                with col1c:
                    st.metric("Convexity", f"{portfolio_metrics['convexity']:.2f}")
    
    with col2:
        st.markdown("**Portfolio Analysis:**")
        
        target_duration = st.number_input("Target Duration", value=5.0, min_value=0.1, max_value=30.0)
        
        if st.button("🎯 Optimize Duration"):
            st.info("Duration optimization would analyze bond weights to match target duration")
        
        if st.button("📊 Risk Analysis"):
            display_portfolio_risk_analysis(portfolio if 'portfolio' in locals() else None)

def parse_bond_portfolio(portfolio_input):
    """Parse bond portfolio input"""
    
    try:
        portfolio = {}
        lines = portfolio_input.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                bond_name, weight = line.strip().split(':')
                portfolio[bond_name.strip()] = float(weight.strip())
        
        return portfolio
        
    except Exception as e:
        return None

def calculate_portfolio_duration_yield(portfolio):
    """Calculate portfolio-level duration and yield"""
    
    try:
        # Sample bond characteristics (in practice, would fetch from database)
        bond_data = {
            'US Treasury 2Y': {'duration': 1.95, 'yield': 4.75, 'convexity': 3.8},
            'Apple 2025': {'duration': 1.2, 'yield': 4.12, 'convexity': 1.5},
            'Microsoft 2024': {'duration': 0.8, 'yield': 3.95, 'convexity': 0.7},
            'JPMorgan 2026': {'duration': 2.8, 'yield': 4.78, 'convexity': 8.2}
        }
        
        total_duration = 0
        total_yield = 0
        total_convexity = 0
        
        for bond_name, weight in portfolio.items():
            if bond_name in bond_data:
                bond_info = bond_data[bond_name]
                total_duration += weight * bond_info['duration']
                total_yield += weight * bond_info['yield']
                total_convexity += weight * bond_info['convexity']
        
        return {
            'duration': total_duration,
            'yield': total_yield,
            'convexity': total_convexity
        }
        
    except Exception as e:
        return None

def display_portfolio_risk_analysis(portfolio):
    """Display portfolio risk analysis"""
    
    if not portfolio:
        st.warning("No portfolio data available for risk analysis")
        return
    
    st.subheader("⚠️ Portfolio Risk Analysis")
    
    # Interest rate shock scenarios
    rate_shocks = [-200, -100, -50, 0, 50, 100, 200]  # basis points
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Interest Rate Shock Analysis:**")
        
        # Calculate portfolio impact for each shock
        portfolio_metrics = calculate_portfolio_duration_yield(portfolio)
        
        if portfolio_metrics:
            shock_results = []
            
            for shock in rate_shocks:
                # Simplified calculation using duration
                shock_decimal = shock / 10000  # Convert bps to decimal
                price_change = -portfolio_metrics['duration'] * shock_decimal * 100  # Percentage
                
                shock_results.append({
                    'Rate Shock (bps)': shock,
                    'Portfolio Impact (%)': price_change
                })
            
            shock_df = pd.DataFrame(shock_results)
            st.dataframe(shock_df, use_container_width=True)
            
            # Shock visualization
            fig_shock = px.bar(
                shock_df,
                x='Rate Shock (bps)',
                y='Portfolio Impact (%)',
                title="Interest Rate Shock Impact"
            )
            
            fig_shock.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_shock, use_container_width=True)
    
    with col2:
        st.markdown("**Duration Contribution Analysis:**")
        
        # Duration contribution by bond
        contributions = []
        
        portfolio_metrics = calculate_portfolio_duration_yield(portfolio)
        
        bond_data = {
            'US Treasury 2Y': {'duration': 1.95},
            'Apple 2025': {'duration': 1.2},
            'Microsoft 2024': {'duration': 0.8},
            'JPMorgan 2026': {'duration': 2.8}
        }
        
        for bond_name, weight in portfolio.items():
            if bond_name in bond_data:
                contribution = weight * bond_data[bond_name]['duration']
                contributions.append({
                    'Bond': bond_name,
                    'Weight': f"{weight:.1%}",
                    'Duration Contribution': f"{contribution:.2f}"
                })
        
        if contributions:
            contrib_df = pd.DataFrame(contributions)
            st.dataframe(contrib_df, use_container_width=True)
        
        # Key risk metrics
        if portfolio_metrics:
            st.markdown("**Key Risk Metrics:**")
            st.metric("DV01 (per $1M)", f"${portfolio_metrics['duration'] * 100:.0f}")
            st.metric("Convexity", f"{portfolio_metrics['convexity']:.2f}")
            
            # Estimate VaR (simplified)
            daily_vol = 0.02  # Estimated daily yield volatility
            var_95 = 1.645 * daily_vol * portfolio_metrics['duration'] * 100
            st.metric("1-Day VaR (95%)", f"{var_95:.2f}%")