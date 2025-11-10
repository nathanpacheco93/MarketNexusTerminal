import requests
import yfinance as yf
import pandas as pd
import streamlit as st
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from bs4 import BeautifulSoup
import re
import logging
from functools import wraps
from threading import Lock
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQuality:
    """Data quality indicators"""
    LIVE = "live"           # Real-time data (<5 seconds old)
    NEAR_REAL_TIME = "near_real_time"  # Near real-time (<60 seconds old)
    DELAYED = "delayed"     # Delayed data (>60 seconds old)
    CACHED = "cached"       # Cached data
    FALLBACK = "fallback"   # Fallback to yfinance
    STALE = "stale"        # Stale data (>15 minutes old)

class MarketDataService:
    """Centralized real-time market data service with fallback to yfinance"""
    
    def __init__(self):
        self._cache = {}
        self._cache_lock = Lock()
        self._rate_limiter = {}
        self._rate_limit_lock = Lock()
        
        # Rate limiting configuration (requests per minute)
        self.rate_limits = {
            'yahoo_finance': 30,    # Conservative limit for scraping
            'finviz': 20,
            'marketwatch': 20,
            'default': 60
        }
        
        # Cache TTL configuration (seconds)
        self.cache_ttl = {
            'stock_quote': 5,       # Stock quotes - 5 seconds
            'index_data': 10,       # Index data - 10 seconds  
            'currency_data': 30,    # Currency data - 30 seconds
            'commodity_data': 30,   # Commodity data - 30 seconds
            'crypto_data': 5,       # Crypto data - 5 seconds
            'company_info': 3600,   # Company info - 1 hour
            'historical_data': 300, # Historical data - 5 minutes
            'options_data': 60,     # Options data - 1 minute
            'futures_data': 15      # Futures data - 15 seconds
        }
        
        # Data source priorities (higher number = higher priority)
        self.data_sources = {
            'yahoo_scrape': 3,      # Yahoo Finance scraping
            'finviz': 2,            # Finviz scraping
            'yfinance': 1           # yfinance fallback
        }

    def _rate_limit_check(self, source: str) -> bool:
        """Check if we can make a request to the source without exceeding rate limits"""
        with self._rate_limit_lock:
            current_time = time.time()
            limit = self.rate_limits.get(source, self.rate_limits['default'])
            
            if source not in self._rate_limiter:
                self._rate_limiter[source] = []
            
            # Remove requests older than 1 minute
            self._rate_limiter[source] = [
                req_time for req_time in self._rate_limiter[source]
                if current_time - req_time < 60
            ]
            
            # Check if we can make another request
            if len(self._rate_limiter[source]) < limit:
                self._rate_limiter[source].append(current_time)
                return True
            
            return False

    def _cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Generate cache key"""
        key_parts = [symbol.upper(), data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}_{v}")
        return "_".join(key_parts)

    def _get_from_cache(self, key: str, ttl: int) -> Optional[Dict]:
        """Get data from cache if not expired"""
        with self._cache_lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                age = time.time() - timestamp
                
                if age < ttl:
                    # Update data quality based on age
                    if age < 5:
                        data['data_quality'] = DataQuality.CACHED
                    elif age < 60:
                        data['data_quality'] = DataQuality.NEAR_REAL_TIME
                    else:
                        data['data_quality'] = DataQuality.DELAYED
                    
                    data['cache_age'] = age
                    return data
                else:
                    # Remove expired data
                    del self._cache[key]
        
        return None

    def _set_cache(self, key: str, data: Dict) -> None:
        """Set data in cache"""
        with self._cache_lock:
            self._cache[key] = (data.copy(), time.time())

    def _scrape_yahoo_quote(self, symbol: str) -> Optional[Dict]:
        """Scrape real-time quote from Yahoo Finance"""
        if not self._rate_limit_check('yahoo_finance'):
            return None
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            url = f"https://finance.yahoo.com/quote/{symbol}"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract price data
            price_element = soup.find('fin-streamer', {'data-symbol': symbol, 'data-field': 'regularMarketPrice'})
            change_element = soup.find('fin-streamer', {'data-symbol': symbol, 'data-field': 'regularMarketChange'})
            change_pct_element = soup.find('fin-streamer', {'data-symbol': symbol, 'data-field': 'regularMarketChangePercent'})
            
            if price_element:
                try:
                    current_price = float(price_element.text.replace(',', ''))
                    change = float(change_element.text.replace(',', '')) if change_element else 0
                    change_percent_text = change_pct_element.text if change_pct_element else "0%"
                    change_percent = float(re.findall(r'[-+]?\d*\.?\d+', change_percent_text)[0])
                    
                    return {
                        'symbol': symbol,
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'data_quality': DataQuality.NEAR_REAL_TIME,
                        'source': 'yahoo_scrape',
                        'timestamp': datetime.now().isoformat(),
                        'market_state': 'REGULAR'  # Could be enhanced to detect pre/post market
                    }
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing Yahoo data for {symbol}: {e}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Yahoo scraping failed for {symbol}: {e}")
            return None

    def _get_yfinance_fallback(self, symbol: str, data_type: str) -> Optional[Dict]:
        """Get data from yfinance as fallback"""
        try:
            ticker = yf.Ticker(symbol)
            
            if data_type == 'stock_quote':
                # Try fast_info first for more recent data
                try:
                    fast_info = ticker.fast_info
                    current_price = fast_info.get('lastPrice')
                    prev_close = fast_info.get('previousClose', current_price)
                    
                    if current_price and current_price > 0:
                        change = current_price - prev_close
                        change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                        
                        return {
                            'symbol': symbol,
                            'price': current_price,
                            'change': change,
                            'change_percent': change_percent,
                            'volume': fast_info.get('regularMarketVolume', 0),
                            'data_quality': DataQuality.FALLBACK,
                            'source': 'yfinance_fast',
                            'timestamp': datetime.now().isoformat()
                        }
                except Exception:
                    pass  # Fall through to history method
                
                # Fallback to history method
                hist = ticker.history(period="2d")
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_percent = (change / prev_price) * 100 if prev_price > 0 else 0
                    
                    return {
                        'symbol': symbol,
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': hist['Volume'].iloc[-1],
                        'data_quality': DataQuality.FALLBACK,
                        'source': 'yfinance_history',
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"yfinance fallback failed for {symbol}: {e}")
            
        return None

    def get_stock_quote(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get real-time stock quote with multiple data source fallback"""
        cache_key = self._cache_key(symbol, 'stock_quote')
        
        # Check cache first unless force refresh
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key, self.cache_ttl['stock_quote'])
            if cached_data:
                return cached_data
        
        # Try data sources in priority order
        data_sources_ordered = sorted(self.data_sources.items(), key=lambda x: x[1], reverse=True)
        
        for source_name, priority in data_sources_ordered:
            try:
                if source_name == 'yahoo_scrape':
                    data = self._scrape_yahoo_quote(symbol)
                elif source_name == 'yfinance':
                    data = self._get_yfinance_fallback(symbol, 'stock_quote')
                else:
                    continue  # Skip unknown sources
                
                if data:
                    # Add metadata
                    data['retrieved_at'] = datetime.now().isoformat()
                    data['cache_age'] = 0
                    
                    # Cache the successful result
                    self._set_cache(cache_key, data)
                    return data
                    
            except Exception as e:
                logger.warning(f"Data source {source_name} failed for {symbol}: {e}")
                continue
        
        # If all sources fail, return None
        logger.error(f"All data sources failed for {symbol}")
        return None

    def get_multiple_quotes(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, Dict]:
        """Get multiple stock quotes efficiently"""
        results = {}
        
        for symbol in symbols:
            quote_data = self.get_stock_quote(symbol, force_refresh)
            if quote_data:
                results[symbol] = quote_data
        
        return results

    def get_index_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get real-time index data"""
        cache_key = self._cache_key(symbol, 'index_data')
        
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key, self.cache_ttl['index_data'])
            if cached_data:
                return cached_data
        
        # For indices, try Yahoo scraping first, then yfinance
        data = self._scrape_yahoo_quote(symbol)
        if not data:
            data = self._get_yfinance_fallback(symbol, 'stock_quote')
        
        if data:
            data['retrieved_at'] = datetime.now().isoformat()
            data['cache_age'] = 0
            self._set_cache(cache_key, data)
        
        return data

    def get_currency_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get real-time currency data"""
        cache_key = self._cache_key(symbol, 'currency_data')
        
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key, self.cache_ttl['currency_data'])
            if cached_data:
                return cached_data
        
        # Currency pairs - use yfinance as primary source for now
        data = self._get_yfinance_fallback(symbol, 'stock_quote')
        
        if data:
            data['retrieved_at'] = datetime.now().isoformat()
            data['cache_age'] = 0
            # Currency data is typically less frequently updated
            data['data_quality'] = DataQuality.NEAR_REAL_TIME
            self._set_cache(cache_key, data)
        
        return data

    def get_commodity_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get real-time commodity data"""
        cache_key = self._cache_key(symbol, 'commodity_data')
        
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key, self.cache_ttl['commodity_data'])
            if cached_data:
                return cached_data
        
        # Try scraping first, then yfinance
        data = self._scrape_yahoo_quote(symbol)
        if not data:
            data = self._get_yfinance_fallback(symbol, 'stock_quote')
        
        if data:
            data['retrieved_at'] = datetime.now().isoformat()
            data['cache_age'] = 0
            self._set_cache(cache_key, data)
        
        return data

    def get_crypto_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get real-time cryptocurrency data"""
        cache_key = self._cache_key(symbol, 'crypto_data')
        
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key, self.cache_ttl['crypto_data'])
            if cached_data:
                return cached_data
        
        # Crypto data - try scraping first for more frequent updates
        data = self._scrape_yahoo_quote(symbol)
        if not data:
            data = self._get_yfinance_fallback(symbol, 'stock_quote')
        
        if data:
            data['retrieved_at'] = datetime.now().isoformat()
            data['cache_age'] = 0
            # Crypto should have better real-time quality
            if data['data_quality'] == DataQuality.FALLBACK:
                data['data_quality'] = DataQuality.NEAR_REAL_TIME
            self._set_cache(cache_key, data)
        
        return data

    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical data (primarily from yfinance)"""
        cache_key = self._cache_key(symbol, 'historical_data', period=period, interval=interval)
        
        # Check cache
        cached_data = self._get_from_cache(cache_key, self.cache_ttl['historical_data'])
        if cached_data and 'data' in cached_data:
            return cached_data['data']
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if not hist.empty:
                # Cache the historical data
                cache_data = {
                    'data': hist,
                    'symbol': symbol,
                    'period': period,
                    'interval': interval,
                    'data_quality': DataQuality.FALLBACK,  # Historical data doesn't need real-time
                    'source': 'yfinance',
                    'retrieved_at': datetime.now().isoformat()
                }
                self._set_cache(cache_key, cache_data)
                return hist
                
        except Exception as e:
            logger.error(f"Historical data fetch failed for {symbol}: {e}")
        
        return None

    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get company information (from yfinance)"""
        cache_key = self._cache_key(symbol, 'company_info')
        
        # Check cache (company info changes infrequently)
        cached_data = self._get_from_cache(cache_key, self.cache_ttl['company_info'])
        if cached_data and 'info' in cached_data:
            return cached_data['info']
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info:
                cache_data = {
                    'info': info,
                    'symbol': symbol,
                    'data_quality': DataQuality.FALLBACK,  # Company info doesn't need real-time
                    'source': 'yfinance',
                    'retrieved_at': datetime.now().isoformat()
                }
                self._set_cache(cache_key, cache_data)
                return info
                
        except Exception as e:
            logger.error(f"Company info fetch failed for {symbol}: {e}")
        
        return None

    def get_options_data(self, symbol: str, expiration: str = None) -> Optional[Dict]:
        """Get options data (from yfinance)"""
        cache_key = self._cache_key(symbol, 'options_data', expiration=expiration or 'all')
        
        cached_data = self._get_from_cache(cache_key, self.cache_ttl['options_data'])
        if cached_data and 'options' in cached_data:
            return cached_data['options']
        
        try:
            ticker = yf.Ticker(symbol)
            
            if expiration:
                options_chain = ticker.option_chain(expiration)
                options_data = {
                    'calls': options_chain.calls,
                    'puts': options_chain.puts,
                    'expiration': expiration
                }
            else:
                # Get all expirations
                expirations = ticker.options
                options_data = {
                    'expirations': expirations
                }
            
            cache_data = {
                'options': options_data,
                'symbol': symbol,
                'data_quality': DataQuality.FALLBACK,
                'source': 'yfinance',
                'retrieved_at': datetime.now().isoformat()
            }
            self._set_cache(cache_key, cache_data)
            return options_data
            
        except Exception as e:
            logger.error(f"Options data fetch failed for {symbol}: {e}")
        
        return None

    def is_market_hours(self) -> bool:
        """Check if it's currently market hours (NYSE)"""
        try:
            import pytz
            ny_tz = pytz.timezone('America/New_York')
            now = datetime.now(ny_tz)
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            if now.weekday() >= 5:  # Weekend
                return False
            
            # Market hours: 9:30 AM to 4:00 PM EST
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception:
            # Fallback: assume market hours during typical business hours
            return 9 <= datetime.now().hour <= 16

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._cache_lock:
            total_entries = len(self._cache)
            total_size = sum(len(str(data)) for data, _ in self._cache.values())
            
            # Count by data type
            type_counts = {}
            for key in self._cache.keys():
                data_type = key.split('_')[1] if '_' in key else 'unknown'
                type_counts[data_type] = type_counts.get(data_type, 0) + 1
            
            return {
                'total_entries': total_entries,
                'total_size_bytes': total_size,
                'type_counts': type_counts,
                'cache_keys': list(self._cache.keys())
            }

    def clear_cache(self, data_type: str = None) -> None:
        """Clear cache - all or specific data type"""
        with self._cache_lock:
            if data_type:
                # Clear specific data type
                keys_to_remove = [key for key in self._cache.keys() if data_type in key]
                for key in keys_to_remove:
                    del self._cache[key]
            else:
                # Clear all cache
                self._cache.clear()

# Global instance
_data_service = None

def get_data_service() -> MarketDataService:
    """Get singleton instance of data service"""
    global _data_service
    if _data_service is None:
        _data_service = MarketDataService()
    return _data_service

# Convenience functions for backward compatibility
def get_real_time_quote(symbol: str, force_refresh: bool = False) -> Optional[Dict]:
    """Get real-time quote for a symbol"""
    return get_data_service().get_stock_quote(symbol, force_refresh)

def get_multiple_real_time_quotes(symbols: List[str], force_refresh: bool = False) -> Dict[str, Dict]:
    """Get multiple real-time quotes"""
    return get_data_service().get_multiple_quotes(symbols, force_refresh)

# Data quality indicator helper functions
def format_data_quality_indicator(quality: str, age: float = 0) -> str:
    """Format data quality indicator for display"""
    indicators = {
        DataQuality.LIVE: "🟢 LIVE",
        DataQuality.NEAR_REAL_TIME: "🟡 NEAR REAL-TIME", 
        DataQuality.DELAYED: "🟠 DELAYED",
        DataQuality.CACHED: "🔵 CACHED",
        DataQuality.FALLBACK: "🔴 DELAYED",
        DataQuality.STALE: "⚫ STALE"
    }
    
    indicator = indicators.get(quality, "❓ UNKNOWN")
    
    if age > 0:
        if age < 60:
            age_str = f" ({age:.0f}s)"
        elif age < 3600:
            age_str = f" ({age/60:.0f}m)"
        else:
            age_str = f" ({age/3600:.1f}h)"
        indicator += age_str
    
    return indicator

def get_data_quality_color(quality: str) -> str:
    """Get color for data quality indicator"""
    colors = {
        DataQuality.LIVE: "green",
        DataQuality.NEAR_REAL_TIME: "orange", 
        DataQuality.DELAYED: "red",
        DataQuality.CACHED: "blue",
        DataQuality.FALLBACK: "red",
        DataQuality.STALE: "gray"
    }
    return colors.get(quality, "gray")

# Streamlit caching decorators that use the new data service
def cache_realtime_data(ttl: int = 5):
    """Decorator for caching real-time data in Streamlit"""
    def decorator(func):
        @wraps(func)
        @st.cache_data(ttl=ttl, show_spinner=False)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator