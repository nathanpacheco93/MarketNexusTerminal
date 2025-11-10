import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def display_risk_management():
    """Display risk management tools and analytics"""
    
    st.subheader("⚠️ Risk Management & Portfolio Analytics")
    
    # Portfolio input section
    st.markdown("### 📊 Portfolio Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        portfolio_input = st.text_area(
            "Enter Portfolio Holdings (Symbol:Weight, one per line)",
            value="AAPL:0.3\nMSFT:0.25\nGOOGL:0.2\nTSLA:0.15\nNVDA:0.1",
            help="Format: SYMBOL:WEIGHT (weights should sum to 1.0)"
        )
        
        # Parse portfolio
        portfolio = parse_portfolio_input(portfolio_input)
        
        if portfolio:
            st.success(f"Portfolio loaded: {len(portfolio)} assets, Total weight: {sum(portfolio.values()):.2%}")
            
            # Display portfolio composition
            portfolio_df = pd.DataFrame(list(portfolio.items()), columns=['Symbol', 'Weight'])
            portfolio_df['Weight'] = portfolio_df['Weight'].apply(lambda x: f"{x:.1%}")
            st.dataframe(portfolio_df, use_container_width=True)
        else:
            st.error("Invalid portfolio format. Please check your input.")
            return
    
    with col2:
        portfolio_value = st.number_input("Portfolio Value ($)", value=100000, min_value=1000, step=1000)
        confidence_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
        time_horizon = st.selectbox("Time Horizon (Days)", [1, 5, 10, 22], index=0)
        
        lookback_period = st.selectbox("Historical Data Period", 
                                     ["1y", "2y", "3y", "5y"], index=1)
        
        # Risk-free rate for calculations
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0) / 100
    
    # Fetch data and calculate metrics
    if st.button("🔄 Calculate Risk Metrics", type="primary"):
        with st.spinner("Fetching data and calculating risk metrics..."):
            data = fetch_portfolio_data(list(portfolio.keys()), lookback_period)
            
            if data is not None and not data.empty:
                # Calculate portfolio returns
                portfolio_returns = calculate_portfolio_returns(data, portfolio)
                
                # Display risk metrics
                display_var_analysis(portfolio_returns, portfolio_value, confidence_level, time_horizon)
                
                # Display correlation analysis
                display_correlation_analysis(data, portfolio)
                
                # Display stress testing
                display_stress_testing(portfolio_returns, portfolio_value, data, portfolio)
                
                # Display portfolio optimization
                display_portfolio_optimization(data, portfolio, risk_free_rate)
                
                # Display risk decomposition
                display_risk_decomposition(data, portfolio)
                
            else:
                st.error("Could not fetch data for the specified portfolio. Please check the symbols.")

def parse_portfolio_input(portfolio_input):
    """Parse portfolio input string into dictionary"""
    
    try:
        portfolio = {}
        lines = portfolio_input.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                symbol, weight = line.strip().split(':')
                portfolio[symbol.strip().upper()] = float(weight.strip())
        
        # Validate weights
        total_weight = sum(portfolio.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            st.warning(f"Portfolio weights sum to {total_weight:.2%}, consider adjusting to 100%")
        
        return portfolio
        
    except Exception as e:
        return None

def fetch_portfolio_data(symbols, period):
    """Fetch historical data for portfolio symbols"""
    
    try:
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # Use Adjusted Close for accurate returns
                data[symbol] = hist['Adj Close']
            else:
                st.warning(f"No data found for {symbol}")
        
        if data:
            # Combine into DataFrame and forward fill missing values
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill').dropna()
            return df
        
        return None
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_portfolio_returns(data, portfolio):
    """Calculate portfolio returns based on weights"""
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate weighted portfolio returns
    portfolio_returns = pd.Series(0, index=returns.index)
    
    for symbol, weight in portfolio.items():
        if symbol in returns.columns:
            portfolio_returns += returns[symbol] * weight
    
    return portfolio_returns

def display_var_analysis(portfolio_returns, portfolio_value, confidence_level, time_horizon):
    """Display Value at Risk analysis"""
    
    st.subheader("📉 Value at Risk (VaR) Analysis")
    
    # Calculate different VaR methods
    historical_var = calculate_historical_var(portfolio_returns, confidence_level, time_horizon)
    parametric_var = calculate_parametric_var(portfolio_returns, confidence_level, time_horizon)
    
    # Convert to dollar amounts
    historical_var_dollar = historical_var * portfolio_value
    parametric_var_dollar = parametric_var * portfolio_value
    
    # Expected Shortfall (Conditional VaR)
    expected_shortfall = calculate_expected_shortfall(portfolio_returns, confidence_level, time_horizon)
    expected_shortfall_dollar = expected_shortfall * portfolio_value
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"Historical VaR ({confidence_level:.0%})", 
            f"${historical_var_dollar:,.0f}",
            f"{historical_var:.2%}"
        )
    
    with col2:
        st.metric(
            f"Parametric VaR ({confidence_level:.0%})", 
            f"${parametric_var_dollar:,.0f}",
            f"{parametric_var:.2%}"
        )
    
    with col3:
        st.metric(
            f"Expected Shortfall ({confidence_level:.0%})", 
            f"${expected_shortfall_dollar:,.0f}",
            f"{expected_shortfall:.2%}"
        )
    
    with col4:
        volatility = portfolio_returns.std() * np.sqrt(252)
        st.metric("Annual Volatility", f"{volatility:.2%}")
    
    # VaR visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Returns Distribution', 'Historical VaR', 'Rolling VaR (30-day)', 'Risk Metrics Timeline'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Returns distribution
    fig.add_trace(
        go.Histogram(
            x=portfolio_returns * 100,
            nbinsx=50,
            name='Daily Returns',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add VaR lines to distribution
    fig.add_vline(
        x=historical_var * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Historical VaR: {historical_var:.2%}",
        row=1, col=1
    )
    
    # Historical VaR timeline
    returns_sorted = portfolio_returns.sort_values()
    var_index = int((1 - confidence_level) * len(returns_sorted))
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(returns_sorted))),
            y=returns_sorted * 100,
            mode='lines',
            name='Sorted Returns',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    fig.add_hline(
        y=historical_var * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR Threshold",
        row=1, col=2
    )
    
    # Rolling VaR
    rolling_var = portfolio_returns.rolling(window=30).quantile(1 - confidence_level) * 100
    
    fig.add_trace(
        go.Scatter(
            x=rolling_var.index,
            y=rolling_var,
            mode='lines',
            name='Rolling VaR',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    # Risk metrics timeline
    rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100
    
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode='lines',
            name='Rolling Volatility',
            line=dict(color='orange')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Value at Risk Analysis",
        template="plotly_dark",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Returns (%)", row=1, col=1)
    fig.update_xaxes(title_text="Percentile", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=1, col=2)
    fig.update_yaxes(title_text="VaR (%)", row=2, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def calculate_historical_var(returns, confidence_level, time_horizon):
    """Calculate historical Value at Risk"""
    
    # Scale returns for time horizon
    scaled_returns = returns * np.sqrt(time_horizon)
    
    # Calculate VaR as the quantile
    var = scaled_returns.quantile(1 - confidence_level)
    
    return abs(var)  # Return as positive value

def calculate_parametric_var(returns, confidence_level, time_horizon):
    """Calculate parametric (normal distribution) Value at Risk"""
    
    # Calculate mean and standard deviation
    mean = returns.mean()
    std = returns.std()
    
    # Scale for time horizon
    mean_scaled = mean * time_horizon
    std_scaled = std * np.sqrt(time_horizon)
    
    # Calculate VaR using normal distribution
    z_score = stats.norm.ppf(1 - confidence_level)
    var = -(mean_scaled + z_score * std_scaled)
    
    return max(var, 0)  # Ensure positive

def calculate_expected_shortfall(returns, confidence_level, time_horizon):
    """Calculate Expected Shortfall (Conditional VaR)"""
    
    # Scale returns for time horizon
    scaled_returns = returns * np.sqrt(time_horizon)
    
    # Calculate VaR threshold
    var_threshold = scaled_returns.quantile(1 - confidence_level)
    
    # Calculate expected shortfall (average of losses beyond VaR)
    tail_losses = scaled_returns[scaled_returns <= var_threshold]
    
    if len(tail_losses) > 0:
        expected_shortfall = abs(tail_losses.mean())
    else:
        expected_shortfall = abs(var_threshold)
    
    return expected_shortfall

def display_correlation_analysis(data, portfolio):
    """Display correlation analysis and heatmap"""
    
    st.subheader("🔗 Correlation Analysis")
    
    # Calculate correlation matrix
    returns = data.pct_change().dropna()
    correlation_matrix = returns.corr()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Portfolio Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto",
            text_auto=True
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation statistics
        st.markdown("**Correlation Statistics:**")
        
        # Average correlation
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        st.metric("Average Correlation", f"{avg_correlation:.3f}")
        
        # Highest and lowest correlations
        corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
        highest_corr = corr_values.max()
        lowest_corr = corr_values.min()
        
        st.metric("Highest Correlation", f"{highest_corr:.3f}")
        st.metric("Lowest Correlation", f"{lowest_corr:.3f}")
        
        # Most correlated pairs
        st.markdown("**Most Correlated Pairs:**")
        
        # Find most correlated pairs
        corr_pairs = []
        symbols = list(correlation_matrix.columns)
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr_pairs.append({
                    'Pair': f"{symbols[i]}-{symbols[j]}",
                    'Correlation': correlation_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
        st.dataframe(corr_df.head(5), use_container_width=True)

def display_stress_testing(portfolio_returns, portfolio_value, data, portfolio):
    """Display stress testing and scenario analysis"""
    
    st.subheader("🚨 Stress Testing & Scenario Analysis")
    
    # Historical stress periods
    stress_scenarios = {
        "2008 Financial Crisis": ("2008-09-01", "2009-03-01"),
        "COVID-19 Crash": ("2020-02-01", "2020-04-01"),
        "Dot-com Bubble": ("2000-03-01", "2001-03-01"),
        "European Debt Crisis": ("2011-07-01", "2011-12-01")
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Historical Stress Scenarios:**")
        
        stress_results = []
        returns = data.pct_change().dropna()
        
        for scenario_name, (start_date, end_date) in stress_scenarios.items():
            try:
                # Filter data for stress period
                mask = (returns.index >= start_date) & (returns.index <= end_date)
                period_returns = returns[mask]
                
                if not period_returns.empty:
                    # Calculate portfolio returns for this period
                    portfolio_period_returns = pd.Series(0, index=period_returns.index)
                    
                    for symbol, weight in portfolio.items():
                        if symbol in period_returns.columns:
                            portfolio_period_returns += period_returns[symbol] * weight
                    
                    # Calculate cumulative return and max drawdown
                    cumulative_return = (1 + portfolio_period_returns).cumprod() - 1
                    max_drawdown = calculate_max_drawdown(cumulative_return)
                    total_return = cumulative_return.iloc[-1]
                    
                    stress_results.append({
                        'Scenario': scenario_name,
                        'Total Return': f"{total_return:.2%}",
                        'Max Drawdown': f"{max_drawdown:.2%}",
                        'Dollar Impact': f"${total_return * portfolio_value:,.0f}"
                    })
                
            except Exception as e:
                continue
        
        if stress_results:
            stress_df = pd.DataFrame(stress_results)
            st.dataframe(stress_df, use_container_width=True)
        else:
            st.info("Historical stress scenarios not available for current data range")
    
    with col2:
        st.markdown("**Monte Carlo Stress Testing:**")
        
        # Monte Carlo simulation
        n_simulations = st.selectbox("Number of Simulations", [100, 500, 1000], index=1)
        simulation_days = st.selectbox("Simulation Period (Days)", [22, 63, 252], index=1)
        
        if st.button("🎲 Run Monte Carlo Simulation"):
            with st.spinner("Running simulations..."):
                mc_results = run_monte_carlo_simulation(
                    portfolio_returns, n_simulations, simulation_days, portfolio_value
                )
                
                # Display simulation results
                var_95 = np.percentile(mc_results, 5)
                var_99 = np.percentile(mc_results, 1)
                expected_value = np.mean(mc_results)
                
                st.metric("Expected Portfolio Value", f"${expected_value:,.0f}")
                st.metric("VaR 95% (MC)", f"${portfolio_value - var_95:,.0f}")
                st.metric("VaR 99% (MC)", f"${portfolio_value - var_99:,.0f}")
                
                # Simulation distribution
                fig = go.Figure()
                
                fig.add_trace(
                    go.Histogram(
                        x=mc_results,
                        nbinsx=50,
                        name='Simulation Results',
                        marker_color='lightgreen',
                        opacity=0.7
                    )
                )
                
                fig.add_vline(x=var_95, line_dash="dash", line_color="orange", 
                             annotation_text="VaR 95%")
                fig.add_vline(x=var_99, line_dash="dash", line_color="red", 
                             annotation_text="VaR 99%")
                
                fig.update_layout(
                    title="Monte Carlo Simulation Results",
                    template="plotly_dark",
                    height=300,
                    xaxis_title="Portfolio Value ($)",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig, use_container_width=True)

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns"""
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / (1 + running_max)
    
    # Return maximum drawdown (most negative value)
    return drawdown.min()

def run_monte_carlo_simulation(portfolio_returns, n_simulations, simulation_days, portfolio_value):
    """Run Monte Carlo simulation for portfolio"""
    
    # Calculate return statistics
    mean_return = portfolio_returns.mean()
    return_std = portfolio_returns.std()
    
    # Run simulations
    simulation_results = []
    
    for _ in range(n_simulations):
        # Generate random returns
        random_returns = np.random.normal(mean_return, return_std, simulation_days)
        
        # Calculate final portfolio value
        final_value = portfolio_value * np.prod(1 + random_returns)
        simulation_results.append(final_value)
    
    return np.array(simulation_results)

def display_portfolio_optimization(data, portfolio, risk_free_rate):
    """Display portfolio optimization analysis"""
    
    st.subheader("⚖️ Portfolio Optimization")
    
    # Calculate expected returns and covariance matrix
    returns = data.pct_change().dropna()
    
    # Annualize returns and covariance
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Current Portfolio Metrics:**")
        
        # Calculate current portfolio metrics
        weights = np.array([portfolio.get(symbol, 0) for symbol in returns.columns])
        
        current_return = np.sum(expected_returns * weights)
        current_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        current_sharpe = (current_return - risk_free_rate) / current_vol if current_vol > 0 else 0
        
        st.metric("Expected Annual Return", f"{current_return:.2%}")
        st.metric("Annual Volatility", f"{current_vol:.2%}")
        st.metric("Sharpe Ratio", f"{current_sharpe:.3f}")
        
        # Asset allocation pie chart
        portfolio_df = pd.DataFrame(list(portfolio.items()), columns=['Symbol', 'Weight'])
        
        fig_pie = px.pie(
            portfolio_df, 
            values='Weight', 
            names='Symbol',
            title="Current Asset Allocation"
        )
        
        fig_pie.update_layout(
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("**Optimization Analysis:**")
        
        if st.button("🎯 Calculate Optimal Portfolios"):
            with st.spinner("Optimizing portfolios..."):
                # Calculate efficient frontier
                optimal_portfolios = calculate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate)
                
                if optimal_portfolios:
                    # Display optimal portfolio metrics
                    min_vol_portfolio = optimal_portfolios['min_vol']
                    max_sharpe_portfolio = optimal_portfolios['max_sharpe']
                    
                    st.markdown("**Minimum Volatility Portfolio:**")
                    st.write(f"Return: {min_vol_portfolio['return']:.2%}")
                    st.write(f"Volatility: {min_vol_portfolio['volatility']:.2%}")
                    st.write(f"Sharpe Ratio: {min_vol_portfolio['sharpe']:.3f}")
                    
                    st.markdown("**Maximum Sharpe Portfolio:**")
                    st.write(f"Return: {max_sharpe_portfolio['return']:.2%}")
                    st.write(f"Volatility: {max_sharpe_portfolio['volatility']:.2%}")
                    st.write(f"Sharpe Ratio: {max_sharpe_portfolio['sharpe']:.3f}")
                    
                    # Efficient frontier plot
                    fig_frontier = go.Figure()
                    
                    # Plot efficient frontier
                    if 'frontier_vols' in optimal_portfolios and 'frontier_returns' in optimal_portfolios:
                        fig_frontier.add_trace(
                            go.Scatter(
                                x=optimal_portfolios['frontier_vols'],
                                y=optimal_portfolios['frontier_returns'],
                                mode='lines',
                                name='Efficient Frontier',
                                line=dict(color='blue', width=2)
                            )
                        )
                    
                    # Plot current portfolio
                    fig_frontier.add_trace(
                        go.Scatter(
                            x=[current_vol],
                            y=[current_return],
                            mode='markers',
                            name='Current Portfolio',
                            marker=dict(color='red', size=10, symbol='star')
                        )
                    )
                    
                    # Plot optimal portfolios
                    fig_frontier.add_trace(
                        go.Scatter(
                            x=[min_vol_portfolio['volatility']],
                            y=[min_vol_portfolio['return']],
                            mode='markers',
                            name='Min Volatility',
                            marker=dict(color='green', size=10)
                        )
                    )
                    
                    fig_frontier.add_trace(
                        go.Scatter(
                            x=[max_sharpe_portfolio['volatility']],
                            y=[max_sharpe_portfolio['return']],
                            mode='markers',
                            name='Max Sharpe',
                            marker=dict(color='orange', size=10)
                        )
                    )
                    
                    fig_frontier.update_layout(
                        title="Efficient Frontier",
                        template="plotly_dark",
                        height=400,
                        xaxis_title="Volatility",
                        yaxis_title="Expected Return"
                    )
                    
                    st.plotly_chart(fig_frontier, use_container_width=True)

def calculate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate):
    """Calculate efficient frontier and optimal portfolios"""
    
    try:
        n_assets = len(expected_returns)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Objective functions
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        def portfolio_return(weights):
            return np.sum(expected_returns * weights)
        
        def negative_sharpe(weights):
            p_return = portfolio_return(weights)
            p_vol = portfolio_volatility(weights)
            return -(p_return - risk_free_rate) / p_vol if p_vol > 0 else -999
        
        # Optimize for minimum volatility
        min_vol_result = minimize(portfolio_volatility, x0, method='SLSQP', 
                                bounds=bounds, constraints=constraints)
        
        # Optimize for maximum Sharpe ratio
        max_sharpe_result = minimize(negative_sharpe, x0, method='SLSQP',
                                   bounds=bounds, constraints=constraints)
        
        if min_vol_result.success and max_sharpe_result.success:
            min_vol_weights = min_vol_result.x
            max_sharpe_weights = max_sharpe_result.x
            
            return {
                'min_vol': {
                    'weights': min_vol_weights,
                    'return': portfolio_return(min_vol_weights),
                    'volatility': portfolio_volatility(min_vol_weights),
                    'sharpe': (portfolio_return(min_vol_weights) - risk_free_rate) / portfolio_volatility(min_vol_weights)
                },
                'max_sharpe': {
                    'weights': max_sharpe_weights,
                    'return': portfolio_return(max_sharpe_weights),
                    'volatility': portfolio_volatility(max_sharpe_weights),
                    'sharpe': (portfolio_return(max_sharpe_weights) - risk_free_rate) / portfolio_volatility(max_sharpe_weights)
                }
            }
        
        return None
        
    except Exception as e:
        st.error(f"Error in portfolio optimization: {str(e)}")
        return None

def display_risk_decomposition(data, portfolio):
    """Display risk decomposition and contribution analysis"""
    
    st.subheader("📈 Risk Decomposition")
    
    # Calculate component contributions
    returns = data.pct_change().dropna()
    
    # Calculate individual asset volatilities
    asset_vols = returns.std() * np.sqrt(252)
    
    # Calculate portfolio volatility
    portfolio_returns = calculate_portfolio_returns(data, portfolio)
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    
    # Risk contribution analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Asset Risk Contributions:**")
        
        risk_contributions = []
        
        for symbol in portfolio.keys():
            if symbol in returns.columns:
                weight = portfolio[symbol]
                asset_vol = asset_vols[symbol]
                
                # Calculate marginal contribution to risk
                # This is a simplified approach
                marginal_contribution = weight * asset_vol
                risk_contribution = marginal_contribution / portfolio_vol if portfolio_vol > 0 else 0
                
                risk_contributions.append({
                    'Asset': symbol,
                    'Weight': f"{weight:.1%}",
                    'Volatility': f"{asset_vol:.2%}",
                    'Risk Contribution': f"{risk_contribution:.2%}"
                })
        
        if risk_contributions:
            risk_df = pd.DataFrame(risk_contributions)
            st.dataframe(risk_df, use_container_width=True)
            
            # Risk contribution chart
            fig_risk = px.bar(
                risk_df,
                x='Asset',
                y=[float(x.strip('%')) for x in risk_df['Risk Contribution']],
                title="Risk Contribution by Asset"
            )
            
            fig_risk.update_layout(
                template="plotly_dark",
                height=300,
                yaxis_title="Risk Contribution (%)"
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        st.markdown("**Portfolio Risk Metrics:**")
        
        # Additional risk metrics
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        
        # Sharpe ratio
        annual_return = portfolio_returns.mean() * 252
        sharpe_ratio = annual_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        max_dd = calculate_max_drawdown(cumulative_returns)
        
        # Display metrics
        metrics_data = {
            'Metric': ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio', 
                      'Skewness', 'Kurtosis', 'Max Drawdown'],
            'Value': [f"{annual_return:.2%}", f"{portfolio_vol:.2%}", f"{sharpe_ratio:.3f}", 
                     f"{sortino_ratio:.3f}", f"{skewness:.3f}", f"{kurtosis:.3f}", f"{max_dd:.2%}"]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Risk-Return scatter
        fig_scatter = go.Figure()
        
        for symbol in portfolio.keys():
            if symbol in returns.columns:
                asset_return = returns[symbol].mean() * 252
                asset_vol = asset_vols[symbol]
                weight = portfolio[symbol]
                
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[asset_vol],
                        y=[asset_return],
                        mode='markers+text',
                        name=symbol,
                        text=[symbol],
                        textposition="top center",
                        marker=dict(size=weight*500, opacity=0.7)  # Size proportional to weight
                    )
                )
        
        # Add portfolio point
        fig_scatter.add_trace(
            go.Scatter(
                x=[portfolio_vol],
                y=[annual_return],
                mode='markers+text',
                name='Portfolio',
                text=['Portfolio'],
                textposition="top center",
                marker=dict(color='red', size=15, symbol='star')
            )
        )
        
        fig_scatter.update_layout(
            title="Risk-Return Profile",
            template="plotly_dark",
            height=400,
            xaxis_title="Volatility",
            yaxis_title="Expected Return",
            showlegend=False
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)