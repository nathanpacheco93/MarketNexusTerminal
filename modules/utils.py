import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def format_currency(value, currency="USD"):
    """Format currency values"""
    if currency == "USD":
        return f"${value:,.2f}"
    elif currency == "EUR":
        return f"€{value:,.2f}"
    elif currency == "GBP":
        return f"£{value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

def format_percentage(value):
    """Format percentage values"""
    return f"{value:.2f}%"

def format_volume(value):
    """Format volume values"""
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.0f}"

def get_color_for_change(change):
    """Get color for price changes"""
    if change > 0:
        return "green"
    elif change < 0:
        return "red"
    else:
        return "gray"

def calculate_returns(prices):
    """Calculate returns from price series"""
    return prices.pct_change().dropna()

def calculate_volatility(returns, periods=252):
    """Calculate annualized volatility"""
    return returns.std() * np.sqrt(periods)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods=252):
    """Calculate Sharpe ratio"""
    excess_returns = returns.mean() * periods - risk_free_rate
    volatility = calculate_volatility(returns, periods)
    return excess_returns / volatility if volatility != 0 else 0

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def get_trading_days_between(start_date, end_date):
    """Get number of trading days between two dates"""
    # Simple approximation - excludes weekends
    total_days = (end_date - start_date).days
    weeks = total_days // 7
    remaining_days = total_days % 7
    
    # Rough estimate excluding weekends
    trading_days = weeks * 5 + max(0, min(remaining_days, 5))
    return trading_days

def validate_symbol(symbol):
    """Validate if a stock symbol exists"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return 'symbol' in info or 'shortName' in info
    except:
        return False

def get_market_status():
    """Get current market status"""
    now = datetime.now()
    
    # Simple market hours check (US Eastern Time approximation)
    if now.weekday() < 5:  # Monday to Friday
        if 9 <= now.hour < 16:  # 9 AM to 4 PM (simplified)
            return "🟢 Market Open"
        else:
            return "🔴 Market Closed"
    else:
        return "🔴 Market Closed (Weekend)"

def calculate_technical_indicators(data):
    """Calculate common technical indicators"""
    indicators = {}
    
    if len(data) >= 14:
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs)).iloc[-1]
    
    if len(data) >= 20:
        # Bollinger Bands
        sma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = (sma20 + 2 * std20).iloc[-1]
        indicators['BB_Lower'] = (sma20 - 2 * std20).iloc[-1]
        indicators['BB_Middle'] = sma20.iloc[-1]
    
    if len(data) >= 50:
        # Moving averages
        indicators['SMA_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
        indicators['SMA_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
    
    return indicators

def format_market_cap(market_cap):
    """Format market capitalization"""
    if market_cap >= 1e12:
        return f"${market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"${market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"${market_cap/1e6:.2f}M"
    else:
        return f"${market_cap:,.0f}"

def get_sector_performance():
    """Get sector performance data"""
    sector_etfs = {
        "Technology": "XLK",
        "Healthcare": "XLV", 
        "Financials": "XLF",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Communication": "XLC"
    }
    
    sector_data = []
    
    for sector, etf in sector_etfs.items():
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(period="2d")
            
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change_percent = ((current_price - prev_price) / prev_price) * 100
                
                sector_data.append({
                    'Sector': sector,
                    'ETF': etf,
                    'Change %': change_percent
                })
        except:
            continue
    
    return sorted(sector_data, key=lambda x: x['Change %'], reverse=True)

def export_data_to_csv(data, filename):
    """Export data to CSV format"""
    if isinstance(data, pd.DataFrame):
        return data.to_csv(index=False)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    else:
        return str(data)

def get_risk_metrics(returns):
    """Calculate risk metrics for a return series"""
    if len(returns) < 2:
        return {}
    
    metrics = {
        'Mean Return': returns.mean(),
        'Volatility': returns.std(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Min Return': returns.min(),
        'Max Return': returns.max()
    }
    
    return metrics

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_currency_rates():
    """Get major currency exchange rates"""
    currencies = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
    rates = {}
    
    for currency in currencies:
        try:
            ticker = yf.Ticker(currency)
            hist = ticker.history(period="1d")
            if not hist.empty:
                rates[currency.replace('=X', '')] = hist['Close'].iloc[-1]
        except:
            continue
    
    return rates

def display_disclaimer():
    """Display financial disclaimer"""
    st.markdown("""
    ---
    **⚠️ Important Disclaimer:**
    
    This application is for educational and informational purposes only. 
    The data and analysis provided should not be considered as financial advice. 
    Always consult with qualified financial professionals before making investment decisions.
    
    *Data sources: Yahoo Finance and other public APIs*
    """)
