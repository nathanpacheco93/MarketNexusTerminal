import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def display_backtesting():
    """Display backtesting framework for trading strategies"""
    
    st.subheader("🔄 Strategy Backtesting Framework")
    
    # Strategy selection and configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**🎯 Strategy Configuration**")
        
        strategy_type = st.selectbox(
            "Select Strategy Type",
            ["Simple Moving Average Crossover", "RSI Mean Reversion", "Bollinger Bands Bounce", 
             "MACD Signal", "Buy and Hold", "Custom Strategy"]
        )
        
        symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, SPY")
        
        col1a, col1b = st.columns(2)
        with col1a:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
        with col1b:
            end_date = st.date_input("End Date", value=datetime.now())
        
        initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)
        
    with col2:
        st.markdown("**⚙️ Strategy Parameters**")
        
        if strategy_type == "Simple Moving Average Crossover":
            short_window = st.number_input("Short MA Period", value=20, min_value=5, max_value=200)
            long_window = st.number_input("Long MA Period", value=50, min_value=10, max_value=300)
            if short_window >= long_window:
                st.warning("Short MA period should be less than Long MA period")
        
        elif strategy_type == "RSI Mean Reversion":
            rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=50)
            rsi_oversold = st.number_input("Oversold Threshold", value=30, min_value=10, max_value=40)
            rsi_overbought = st.number_input("Overbought Threshold", value=70, min_value=60, max_value=90)
        
        elif strategy_type == "Bollinger Bands Bounce":
            bb_period = st.number_input("Bollinger Bands Period", value=20, min_value=10, max_value=50)
            bb_std = st.number_input("Standard Deviations", value=2.0, min_value=1.0, max_value=3.0, step=0.1)
        
        elif strategy_type == "MACD Signal":
            macd_fast = st.number_input("MACD Fast Period", value=12, min_value=5, max_value=30)
            macd_slow = st.number_input("MACD Slow Period", value=26, min_value=20, max_value=50)
            macd_signal = st.number_input("MACD Signal Period", value=9, min_value=5, max_value=20)
        
        # Risk management parameters
        st.markdown("**⚠️ Risk Management**")
        stop_loss = st.number_input("Stop Loss (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.5) / 100
        take_profit = st.number_input("Take Profit (%)", value=10.0, min_value=0.0, max_value=50.0, step=0.5) / 100
        max_position_size = st.number_input("Max Position Size (%)", value=100, min_value=10, max_value=100, step=5) / 100
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary"):
        if symbol and start_date < end_date:
            with st.spinner(f"Running backtest for {strategy_type} on {symbol}..."):
                results = run_backtest(
                    symbol, start_date, end_date, strategy_type, initial_capital,
                    {
                        'short_window': locals().get('short_window', 20),
                        'long_window': locals().get('long_window', 50),
                        'rsi_period': locals().get('rsi_period', 14),
                        'rsi_oversold': locals().get('rsi_oversold', 30),
                        'rsi_overbought': locals().get('rsi_overbought', 70),
                        'bb_period': locals().get('bb_period', 20),
                        'bb_std': locals().get('bb_std', 2.0),
                        'macd_fast': locals().get('macd_fast', 12),
                        'macd_slow': locals().get('macd_slow', 26),
                        'macd_signal': locals().get('macd_signal', 9),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'max_position_size': max_position_size
                    }
                )
                
                if results is not None:
                    display_backtest_results(results)
                else:
                    st.error("Failed to run backtest. Please check the symbol and try again.")
        else:
            st.error("Please enter a valid symbol and ensure start date is before end date.")

def run_backtest(symbol, start_date, end_date, strategy_type, initial_capital, params):
    """Run backtest for specified strategy"""
    
    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
        
        # Calculate strategy signals
        if strategy_type == "Simple Moving Average Crossover":
            signals = calculate_ma_crossover_signals(data, params['short_window'], params['long_window'])
        elif strategy_type == "RSI Mean Reversion":
            signals = calculate_rsi_signals(data, params['rsi_period'], params['rsi_oversold'], params['rsi_overbought'])
        elif strategy_type == "Bollinger Bands Bounce":
            signals = calculate_bb_signals(data, params['bb_period'], params['bb_std'])
        elif strategy_type == "MACD Signal":
            signals = calculate_macd_signals(data, params['macd_fast'], params['macd_slow'], params['macd_signal'])
        elif strategy_type == "Buy and Hold":
            signals = calculate_buy_hold_signals(data)
        else:
            signals = calculate_buy_hold_signals(data)  # Default
        
        # Add signals to data
        data = data.join(signals)
        
        # Calculate positions and returns
        portfolio = calculate_portfolio_performance(
            data, initial_capital, params['stop_loss'], params['take_profit'], params['max_position_size']
        )
        
        # Calculate metrics
        metrics = calculate_performance_metrics(portfolio, data)
        
        return {
            'symbol': symbol,
            'strategy': strategy_type,
            'data': data,
            'portfolio': portfolio,
            'metrics': metrics,
            'params': params,
            'initial_capital': initial_capital
        }
        
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        return None

def calculate_ma_crossover_signals(data, short_window, long_window):
    """Calculate moving average crossover signals"""
    
    signals = pd.DataFrame(index=data.index)
    
    # Calculate moving averages
    signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
    signals['long_ma'] = data['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(
        signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1, 0
    )
    
    # Calculate positions (1 for long, 0 for no position, -1 for short)
    signals['position'] = signals['signal'].diff()
    
    return signals

def calculate_rsi_signals(data, period, oversold, overbought):
    """Calculate RSI mean reversion signals"""
    
    signals = pd.DataFrame(index=data.index)
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    signals['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    signals['signal'] = 0
    signals.loc[signals['rsi'] < oversold, 'signal'] = 1  # Buy when oversold
    signals.loc[signals['rsi'] > overbought, 'signal'] = -1  # Sell when overbought
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def calculate_bb_signals(data, period, std_dev):
    """Calculate Bollinger Bands bounce signals"""
    
    signals = pd.DataFrame(index=data.index)
    
    # Calculate Bollinger Bands
    signals['middle'] = data['Close'].rolling(window=period).mean()
    signals['std'] = data['Close'].rolling(window=period).std()
    signals['upper'] = signals['middle'] + (signals['std'] * std_dev)
    signals['lower'] = signals['middle'] - (signals['std'] * std_dev)
    
    # Generate signals
    signals['signal'] = 0
    signals.loc[data['Close'] < signals['lower'], 'signal'] = 1  # Buy when below lower band
    signals.loc[data['Close'] > signals['upper'], 'signal'] = -1  # Sell when above upper band
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def calculate_macd_signals(data, fast_period, slow_period, signal_period):
    """Calculate MACD signals"""
    
    signals = pd.DataFrame(index=data.index)
    
    # Calculate MACD
    ema_fast = data['Close'].ewm(span=fast_period).mean()
    ema_slow = data['Close'].ewm(span=slow_period).mean()
    signals['macd'] = ema_fast - ema_slow
    signals['signal_line'] = signals['macd'].ewm(span=signal_period).mean()
    signals['histogram'] = signals['macd'] - signals['signal_line']
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][1:] = np.where(
        (signals['macd'][1:] > signals['signal_line'][1:]) & 
        (signals['macd'][:-1].values <= signals['signal_line'][:-1].values), 1, 0
    )
    signals['signal'][1:] = np.where(
        (signals['macd'][1:] < signals['signal_line'][1:]) & 
        (signals['macd'][:-1].values >= signals['signal_line'][:-1].values), -1, signals['signal'][1:]
    )
    
    # Calculate positions
    signals['position'] = signals['signal'].diff()
    
    return signals

def calculate_buy_hold_signals(data):
    """Calculate buy and hold signals"""
    
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0  # Initialize as no signal
    signals['position'] = 0
    
    # Generate buy signal on the first day
    if len(signals) > 0:
        signals.iloc[0, signals.columns.get_loc('signal')] = 1
        signals.iloc[0, signals.columns.get_loc('position')] = 1
    
    return signals

def calculate_portfolio_performance(data, initial_capital, stop_loss, take_profit, max_position_size):
    """Calculate portfolio performance with risk management and proper execution timing"""
    
    portfolio = pd.DataFrame(index=data.index)
    
    # Use Adjusted Close for accurate pricing (accounts for splits/dividends)
    if 'Adj Close' in data.columns:
        portfolio['price'] = data['Adj Close']
    else:
        portfolio['price'] = data['Close']
    
    # Shift signals by 1 to avoid look-ahead bias (trade on next bar)
    portfolio['signal'] = data['signal'].shift(1).fillna(0)
    portfolio['position'] = data['position'].shift(1).fillna(0)
    
    # Initialize portfolio values
    portfolio['holdings'] = 0.0
    portfolio['cash'] = float(initial_capital)
    portfolio['total'] = float(initial_capital)
    portfolio['returns'] = 0.0
    
    # Track entry prices for stop loss/take profit and trades
    entry_price = 0.0
    current_position = 0
    trade_log = []  # Track actual trades
    
    # Transaction costs (0.1% per trade)
    transaction_cost = 0.001
    
    for i in range(1, len(portfolio)):
        current_price = portfolio['price'].iloc[i]
        
        # Check for new signals
        if portfolio['position'].iloc[i] != 0:
            # New position signal
            if current_position == 0:  # Not currently in position
                position_value = portfolio['cash'].iloc[i-1] * max_position_size
                
                if portfolio['position'].iloc[i] > 0:  # Buy signal
                    # Calculate shares and costs
                    gross_cost = position_value
                    total_cost = gross_cost * (1 + transaction_cost)
                    
                    if total_cost <= portfolio['cash'].iloc[i-1]:
                        shares_to_buy = gross_cost / current_price
                        portfolio.loc[portfolio.index[i], 'holdings'] = shares_to_buy
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - total_cost
                        entry_price = current_price
                        current_position = 1
                        
                        # Log trade
                        trade_log.append({
                            'date': portfolio.index[i],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': gross_cost,
                            'cost': total_cost - gross_cost,
                            'reason': 'SIGNAL'
                        })
                    else:
                        # Insufficient cash
                        portfolio.loc[portfolio.index[i], 'holdings'] = portfolio['holdings'].iloc[i-1]
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                
                elif portfolio['position'].iloc[i] < 0:  # Sell signal (close position)
                    if current_position == 1:
                        # Close long position
                        shares = portfolio['holdings'].iloc[i-1]
                        gross_value = shares * current_price
                        transaction_fee = gross_value * transaction_cost
                        net_value = gross_value - transaction_fee
                        
                        portfolio.loc[portfolio.index[i], 'holdings'] = 0
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + net_value
                        
                        # Log trade
                        trade_log.append({
                            'date': portfolio.index[i],
                            'action': 'SELL',
                            'price': current_price,
                            'shares': shares,
                            'value': gross_value,
                            'cost': transaction_fee,
                            'reason': 'SIGNAL',
                            'pnl': net_value - (shares * entry_price * (1 + transaction_cost))
                        })
                        
                        current_position = 0
                        entry_price = 0
                    else:
                        # No position to close
                        portfolio.loc[portfolio.index[i], 'holdings'] = portfolio['holdings'].iloc[i-1]
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
        else:
            # No new signal, carry forward positions
            portfolio.loc[portfolio.index[i], 'holdings'] = portfolio['holdings'].iloc[i-1]
            portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
        
        # Check stop loss and take profit
        if current_position == 1 and entry_price > 0:
            price_change = (current_price - entry_price) / entry_price
            
            if price_change <= -stop_loss or price_change >= take_profit:
                # Trigger stop loss or take profit
                shares = portfolio['holdings'].iloc[i]
                gross_value = shares * current_price
                transaction_fee = gross_value * transaction_cost
                net_value = gross_value - transaction_fee
                
                portfolio.loc[portfolio.index[i], 'holdings'] = 0
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i] + net_value
                
                # Log trade
                reason = 'STOP_LOSS' if price_change <= -stop_loss else 'TAKE_PROFIT'
                trade_log.append({
                    'date': portfolio.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'value': gross_value,
                    'cost': transaction_fee,
                    'reason': reason,
                    'pnl': net_value - (shares * entry_price * (1 + transaction_cost))
                })
                
                current_position = 0
                entry_price = 0
        
        # Calculate total portfolio value
        portfolio.loc[portfolio.index[i], 'total'] = (
            portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i] * current_price
        )
        
        # Calculate returns
        portfolio.loc[portfolio.index[i], 'returns'] = (
            (portfolio['total'].iloc[i] / portfolio['total'].iloc[i-1]) - 1
        )
    
    # Store trade log in portfolio for analysis
    portfolio.attrs['trade_log'] = trade_log
    
    return portfolio

def calculate_performance_metrics(portfolio, data):
    """Calculate comprehensive performance metrics"""
    
    # Basic returns
    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
    
    # Annualized return
    days = (portfolio.index[-1] - portfolio.index[0]).days
    annualized_return = (1 + total_return) ** (365.25 / days) - 1
    
    # Volatility (annualized)
    daily_returns = portfolio['returns'].dropna()
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    # Sharpe ratio (proper calculation using daily excess returns)
    risk_free_rate = 0.05
    daily_rf = risk_free_rate / 252  # Daily risk-free rate
    excess_returns = daily_returns - daily_rf
    sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
    
    # Max drawdown
    rolling_max = portfolio['total'].cummax()
    drawdown = (portfolio['total'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Trade-based metrics using actual trade log
    trade_log = portfolio.attrs.get('trade_log', [])
    
    # Calculate actual trade statistics
    if trade_log:
        # Group buy/sell pairs
        trade_pairs = []
        buy_trades = [t for t in trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in trade_log if t['action'] == 'SELL']
        
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy = buy_trades[i]
                trade_pairs.append({
                    'entry_date': buy['date'],
                    'exit_date': sell['date'],
                    'entry_price': buy['price'],
                    'exit_price': sell['price'],
                    'shares': buy['shares'],
                    'pnl': sell.get('pnl', 0),
                    'return': (sell['price'] - buy['price']) / buy['price'],
                    'duration': (sell['date'] - buy['date']).days,
                    'exit_reason': sell['reason']
                })
        
        if trade_pairs:
            winning_trades = sum(1 for t in trade_pairs if t['pnl'] > 0)
            total_trades = len(trade_pairs)
            win_rate = winning_trades / total_trades
            
            avg_win = np.mean([t['pnl'] for t in trade_pairs if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in trade_pairs if t['pnl'] < 0]) if (total_trades - winning_trades) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            avg_trade_duration = np.mean([t['duration'] for t in trade_pairs])
        else:
            total_trades = 0
            win_rate = 0
            profit_factor = 0
            avg_trade_duration = 0
    else:
        total_trades = 0
        win_rate = 0
        profit_factor = 0
        avg_trade_duration = 0
    
    # Buy and hold comparison (using adjusted close if available)
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    buy_hold_return = (data[price_col].iloc[-1] / data[price_col].iloc[0]) - 1
    buy_hold_annualized = (1 + buy_hold_return) ** (365.25 / days) - 1
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Exposure (time in market)
    exposure = (portfolio['holdings'] > 0).sum() / len(portfolio)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'profit_factor': profit_factor,
        'avg_trade_duration': avg_trade_duration,
        'exposure': exposure,
        'buy_hold_return': buy_hold_return,
        'buy_hold_annualized': buy_hold_annualized,
        'calmar_ratio': calmar_ratio,
        'final_value': portfolio['total'].iloc[-1]
    }

def display_backtest_results(results):
    """Display comprehensive backtest results"""
    
    st.subheader(f"📊 Backtest Results: {results['strategy']} on {results['symbol']}")
    
    # Performance metrics
    metrics = results['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return", 
            f"{metrics['total_return']:.2%}",
            f"vs B&H: {(metrics['total_return'] - metrics['buy_hold_return']):.2%}"
        )
        st.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        st.metric("Volatility", f"{metrics['volatility']:.2%}")
    
    with col4:
        st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
        st.metric("Total Trades", f"{metrics['total_trades']}")
    
    # Portfolio value chart
    st.subheader("📈 Portfolio Performance")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Portfolio Value vs Buy & Hold', 'Daily Returns', 'Drawdown'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Portfolio value
    portfolio_data = results['portfolio']
    buy_hold_value = results['initial_capital'] * (results['data']['Close'] / results['data']['Close'].iloc[0])
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data['total'],
            mode='lines',
            name='Strategy',
            line=dict(color='#00ff41', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=buy_hold_value.index,
            y=buy_hold_value.values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Daily returns
    fig.add_trace(
        go.Bar(
            x=portfolio_data.index,
            y=portfolio_data['returns'] * 100,
            name='Daily Returns (%)',
            marker_color=np.where(portfolio_data['returns'] > 0, '#00ff41', '#ff6b6b'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Drawdown
    rolling_max = portfolio_data['total'].cummax()
    drawdown = ((portfolio_data['total'] - rolling_max) / rolling_max) * 100
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown (%)',
            line=dict(color='#ff6b6b'),
            fillcolor='rgba(255, 107, 107, 0.3)',
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f"Backtest Analysis: {results['strategy']}",
        template="plotly_dark",
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade analysis
    if 'position' in results['data'].columns:
        trades = analyze_trades(results['portfolio'], results['data'])
        if trades is not None and len(trades) > 0:
            st.subheader("📝 Trade Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Recent Trades:**")
                trade_display = trades[['entry_date', 'exit_date', 'return', 'duration']].tail(10)
                trade_display['return'] = trade_display['return'].apply(lambda x: f"{x:.2%}")
                trade_display['duration'] = trade_display['duration'].apply(lambda x: f"{x} days")
                st.dataframe(trade_display, use_container_width=True)
            
            with col2:
                st.markdown("**Trade Distribution:**")
                
                # Returns distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(
                    go.Histogram(
                        x=trades['return'] * 100,
                        nbinsx=20,
                        name='Trade Returns',
                        marker_color='#00ff41',
                        opacity=0.7
                    )
                )
                
                fig_dist.update_layout(
                    title="Trade Returns Distribution",
                    template="plotly_dark",
                    height=300,
                    xaxis_title="Return (%)",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # Strategy-specific analysis
    display_strategy_analysis(results)

def analyze_trades(portfolio, data):
    """Analyze individual trades using actual trade log"""
    
    try:
        # Use the actual trade log from portfolio execution
        trade_log = portfolio.attrs.get('trade_log', [])
        
        if not trade_log:
            return None
        
        # Group buy/sell pairs
        trades = []
        buy_trades = [t for t in trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in trade_log if t['action'] == 'SELL']
        
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy = buy_trades[i]
                trades.append({
                    'entry_date': buy['date'],
                    'exit_date': sell['date'],
                    'entry_price': buy['price'],
                    'exit_price': sell['price'],
                    'shares': buy['shares'],
                    'pnl': sell.get('pnl', 0),
                    'return': (sell['price'] - buy['price']) / buy['price'],
                    'duration': (sell['date'] - buy['date']).days,
                    'exit_reason': sell['reason']
                })
        
        return pd.DataFrame(trades) if trades else None
        
    except Exception as e:
        return None

def display_strategy_analysis(results):
    """Display strategy-specific analysis"""
    
    st.subheader("🔍 Strategy Analysis")
    
    strategy = results['strategy']
    data = results['data']
    
    if strategy == "Simple Moving Average Crossover":
        # MA analysis
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='white', width=1)
            )
        )
        
        if 'short_ma' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['short_ma'],
                    mode='lines',
                    name=f"Short MA ({results['params']['short_window']})",
                    line=dict(color='#00ff41', width=1)
                )
            )
        
        if 'long_ma' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['long_ma'],
                    mode='lines',
                    name=f"Long MA ({results['params']['long_window']})",
                    line=dict(color='#ff6b6b', width=1)
                )
            )
        
        # Add buy/sell signals
        buy_signals = data[data['position'] > 0]
        sell_signals = data[data['position'] < 0]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='#00ff41', symbol='triangle-up', size=10)
                )
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='#ff6b6b', symbol='triangle-down', size=10)
                )
            )
        
        fig.update_layout(
            title="Moving Average Crossover Strategy",
            template="plotly_dark",
            height=400,
            xaxis_title="Date",
            yaxis_title="Price ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif strategy == "RSI Mean Reversion":
        # RSI analysis
        if 'rsi' in data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price & Signals', 'RSI'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price', line=dict(color='white')),
                row=1, col=1
            )
            
            # Buy/sell signals
            buy_signals = data[data['position'] > 0]
            sell_signals = data[data['position'] < 0]
            
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index, y=buy_signals['Close'], mode='markers',
                        name='Buy Signal', marker=dict(color='#00ff41', symbol='triangle-up', size=8)
                    ),
                    row=1, col=1
                )
            
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index, y=sell_signals['Close'], mode='markers',
                        name='Sell Signal', marker=dict(color='#ff6b6b', symbol='triangle-down', size=8)
                    ),
                    row=1, col=1
                )
            
            # RSI chart
            fig.add_trace(
                go.Scatter(x=data.index, y=data['rsi'], mode='lines', name='RSI', line=dict(color='orange')),
                row=2, col=1
            )
            
            # RSI levels
            fig.add_hline(y=results['params']['rsi_overbought'], line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=results['params']['rsi_oversold'], line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="solid", line_color="gray", row=2, col=1)
            
            fig.update_layout(
                title="RSI Mean Reversion Strategy",
                template="plotly_dark",
                height=500
            )
            
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Parameter sensitivity analysis
    if st.button("📊 Run Parameter Sensitivity Analysis"):
        with st.spinner("Running sensitivity analysis..."):
            sensitivity_results = run_sensitivity_analysis(results)
            if sensitivity_results:
                display_sensitivity_analysis(sensitivity_results)

def run_sensitivity_analysis(base_results):
    """Run parameter sensitivity analysis"""
    
    # This is a simplified version - in practice, you'd vary key parameters
    # and show how performance changes
    
    strategy = base_results['strategy']
    symbol = base_results['symbol']
    
    # For demonstration, we'll just show the concept
    # In a full implementation, you'd vary parameters and re-run backtests
    
    return {
        'strategy': strategy,
        'base_sharpe': base_results['metrics']['sharpe_ratio'],
        'base_return': base_results['metrics']['annualized_return'],
        'parameter_ranges': {
            'short_window': [10, 15, 20, 25, 30] if strategy == "Simple Moving Average Crossover" else [],
            'long_window': [40, 45, 50, 55, 60] if strategy == "Simple Moving Average Crossover" else []
        }
    }

def display_sensitivity_analysis(sensitivity_results):
    """Display parameter sensitivity analysis results"""
    
    st.subheader("🎛️ Parameter Sensitivity Analysis")
    
    # This would show how different parameter values affect performance
    st.info("Parameter sensitivity analysis helps optimize strategy parameters by showing how performance changes with different values.")
    
    # In a full implementation, this would show:
    # - Heatmaps of parameter combinations vs performance
    # - Optimization results
    # - Robustness analysis
    
    st.markdown("""
    **Key Areas for Sensitivity Analysis:**
    - **Moving Average Periods**: Test different short/long window combinations
    - **RSI Thresholds**: Optimize oversold/overbought levels
    - **Risk Management**: Test different stop-loss and take-profit levels
    - **Position Sizing**: Analyze impact of different position sizes
    """)
    
    # Example parameter impact (simplified)
    if sensitivity_results['strategy'] == "Simple Moving Average Crossover":
        st.markdown(f"""
        **Current Performance:**
        - Sharpe Ratio: {sensitivity_results['base_sharpe']:.2f}
        - Annualized Return: {sensitivity_results['base_return']:.2%}
        
        **Parameter Optimization Suggestions:**
        - Test shorter MA periods for more responsive signals
        - Consider longer periods for trend-following in volatile markets
        - Optimize stop-loss levels based on average true range
        """)