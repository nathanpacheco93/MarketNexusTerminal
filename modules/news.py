import streamlit as st
import feedparser
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from bs4 import BeautifulSoup
import json
import re
from textblob import TextBlob
from typing import Dict, List, Optional, Any
import time
import urllib.parse
from modules.user_auth import user_auth
from modules.database import db_manager
import yfinance as yf

# Enhanced news sources with better RSS feeds and fallback options
NEWS_SOURCES = {
    "Yahoo Finance": {
        "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "fallback": [
            "https://finance.yahoo.com/rss/topstories",
            "https://feeds.yahoo.com/finance/news"
        ],
        "category": "General"
    },
    "MarketWatch": {
        "rss": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "fallback": [
            "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/"
        ],
        "category": "General"
    },
    "Reuters Business": {
        "rss": "https://feeds.reuters.com/reuters/businessNews",
        "fallback": [
            "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"
        ],
        "category": "General"
    },
    "Bloomberg": {
        "rss": "https://feeds.bloomberg.com/markets/news.rss",
        "fallback": [
            "https://feeds.bloomberg.com/technology/news.rss"
        ],
        "category": "General"
    },
    "Financial Times": {
        "rss": "https://www.ft.com/news-feed",
        "fallback": [
            "https://www.ft.com/rss/home/uk"
        ],
        "category": "General"
    },
    "CNBC": {
        "rss": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "fallback": [
            "https://www.cnbc.com/id/10001147/device/rss/rss.html"
        ],
        "category": "General"
    },
    "Seeking Alpha": {
        "rss": "https://seekingalpha.com/feed.xml",
        "fallback": [],
        "category": "Analysis"
    },
    "Earnings Feed": {
        "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "fallback": [],
        "category": "Earnings"
    }
}

# News categories for filtering
NEWS_CATEGORIES = {
    "All": "all",
    "Breaking News": "breaking",
    "Earnings": "earnings", 
    "Mergers": "merger",
    "Market Analysis": "analysis",
    "Crypto": "crypto",
    "Commodities": "commodity"
}

# Keywords for different categories
CATEGORY_KEYWORDS = {
    "breaking": ["breaking", "urgent", "alert", "developing", "just in"],
    "earnings": ["earnings", "quarterly", "q1", "q2", "q3", "q4", "eps", "revenue", "profit"],
    "merger": ["merger", "acquisition", "deal", "takeover", "buyout", "m&a"],
    "analysis": ["analysis", "outlook", "forecast", "target", "rating", "upgrade", "downgrade"],
    "crypto": ["bitcoin", "crypto", "blockchain", "ethereum", "btc", "eth", "defi"],
    "commodity": ["oil", "gold", "silver", "copper", "energy", "commodity", "crude"]
}

def display_news():
    """Enhanced financial news section with comprehensive features"""
    
    st.subheader("📰 Financial News & Market Intelligence")
    
    # Load user preferences
    user_id = user_auth.get_current_user_id()
    preferred_sources = user_auth.get_user_preference("news", "preferred_sources", list(NEWS_SOURCES.keys())[:3])
    show_sentiment = user_auth.get_user_preference("news", "show_sentiment", True)
    auto_refresh = user_auth.get_user_preference("news", "auto_refresh", False)
    
    # News controls and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_sources = st.multiselect(
            "Select News Sources", 
            list(NEWS_SOURCES.keys()),
            default=preferred_sources if preferred_sources else list(NEWS_SOURCES.keys())[:3],
            help="Choose your preferred news sources"
        )
        
        # Save preference
        if selected_sources != preferred_sources:
            user_auth.auto_save_preference("news", "preferred_sources", selected_sources)
    
    with col2:
        category_filter = st.selectbox("Category Filter", list(NEWS_CATEGORIES.keys()))
    
    with col3:
        auto_refresh_enabled = st.checkbox("Auto Refresh (2 min)", value=auto_refresh)
        if auto_refresh_enabled != auto_refresh:
            user_auth.auto_save_preference("news", "auto_refresh", auto_refresh_enabled)
    
    # Advanced filters
    with st.expander("🔍 Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol_filter = st.text_input("Stock Symbol Filter", placeholder="e.g., AAPL, TSLA")
            
        with col2:
            keyword_filter = st.text_input("Keyword Filter", placeholder="e.g., AI, merger, earnings")
            
        with col3:
            sentiment_filter = st.selectbox("Sentiment Filter", ["All", "Positive", "Negative", "Neutral"])
    
    # News search functionality
    with st.expander("📅 News Search"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_keywords = st.text_input("Search Keywords", placeholder="Search news...")
        
        with col2:
            date_from = st.date_input("From Date", value=datetime.now().date() - timedelta(days=7))
            
        with col3:
            date_to = st.date_input("To Date", value=datetime.now().date())
        
        if st.button("🔍 Search News"):
            if search_keywords:
                search_results = search_news_by_criteria(search_keywords, date_from, date_to, selected_sources)
                if search_results:
                    display_news_items(search_results, show_sentiment)
                else:
                    st.info("No news found matching your search criteria.")
    
    # Fetch and display news
    if selected_sources:
        try:
            # Get news from selected sources
            all_news = []
            
            for source in selected_sources:
                try:
                    news_items = fetch_enhanced_news(source)
                    if news_items:
                        for item in news_items:
                            item['source'] = source
                            all_news.append(item)
                except Exception as e:
                    st.warning(f"Could not fetch news from {source}: {str(e)}")
            
            if all_news:
                # Apply filters
                filtered_news = filter_news_items(
                    all_news, 
                    category_filter, 
                    symbol_filter, 
                    keyword_filter, 
                    sentiment_filter
                )
                
                if filtered_news:
                    # Sort by published date
                    filtered_news.sort(key=lambda x: x.get('published_parsed', datetime.min), reverse=True)
                    
                    # Display news
                    display_news_items(filtered_news, show_sentiment)
                else:
                    st.info("No news matches your current filters.")
            else:
                st.error("Unable to fetch news from any selected sources.")
                display_fallback_news()
                
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            display_fallback_news()
    else:
        st.warning("Please select at least one news source.")
    
    # News alerts section
    display_news_alerts()
    
    # Enhanced market sentiment analysis
    display_enhanced_market_sentiment()

@st.cache_data(ttl=120)  # Cache for 2 minutes
def fetch_enhanced_news(source: str) -> List[Dict]:
    """Enhanced news fetcher with fallback mechanisms and better error handling"""
    
    if source not in NEWS_SOURCES:
        return []
    
    source_config = NEWS_SOURCES[source]
    urls_to_try = [source_config['rss']] + source_config.get('fallback', [])
    
    for url in urls_to_try:
        try:
            # Add timeout and user agent to prevent blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Try to fetch with requests first for better error handling
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
            except:
                # Fallback to direct feedparser
                feed = feedparser.parse(url)
            
            if not hasattr(feed, 'entries') or not feed.entries:
                continue
                
            news_items = []
            
            for entry in feed.entries[:20]:  # Limit to 20 items per source
                try:
                    # Enhanced data extraction
                    item = {
                        'title': clean_text(entry.get('title', '')),
                        'summary': clean_text(entry.get('summary', entry.get('description', ''))),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'published_parsed': parse_date(entry.get('published_parsed', entry.get('published', ''))),
                        'category': determine_category(entry.get('title', '') + ' ' + entry.get('summary', '')),
                        'sentiment': analyze_sentiment(entry.get('title', '') + ' ' + entry.get('summary', '')),
                        'summary_generated': False
                    }
                    
                    # Generate summary for long articles
                    if len(item['summary']) > 500:
                        item['summary'] = generate_summary(item['summary'])
                        item['summary_generated'] = True
                    
                    news_items.append(item)
                    
                except Exception as e:
                    continue  # Skip problematic entries
            
            if news_items:
                return news_items
                
        except Exception as e:
            continue  # Try next URL
    
    return []

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def parse_date(date_input) -> datetime:
    """Parse various date formats to datetime object"""
    if not date_input:
        return datetime.min
    
    if isinstance(date_input, tuple):
        try:
            return datetime(*date_input[:6])
        except:
            return datetime.min
    
    if isinstance(date_input, str):
        try:
            from dateutil import parser
            return parser.parse(date_input)
        except:
            return datetime.min
    
    if isinstance(date_input, datetime):
        return date_input
    
    return datetime.min

def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment of news text using TextBlob"""
    try:
        if not text:
            return {"polarity": 0, "label": "Neutral", "confidence": 0}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        
        confidence = abs(polarity)
        
        return {
            "polarity": polarity,
            "label": label,
            "confidence": confidence
        }
    except:
        return {"polarity": 0, "label": "Neutral", "confidence": 0}

def determine_category(text: str) -> str:
    """Determine news category based on keywords"""
    text_lower = text.lower()
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return "general"

def generate_summary(text: str, max_sentences: int = 3) -> str:
    """Generate a summary of long text"""
    try:
        sentences = text.split('.')
        if len(sentences) <= max_sentences:
            return text
        
        # Simple extractive summarization - take first few sentences
        summary_sentences = sentences[:max_sentences]
        summary = '. '.join(summary_sentences) + '.'
        
        # Ensure summary isn't too long
        if len(summary) > 300:
            summary = summary[:297] + "..."
        
        return summary
    except:
        return text[:300] + "..." if len(text) > 300 else text

def filter_news_items(news_items: List[Dict], category_filter: str, symbol_filter: str, 
                     keyword_filter: str, sentiment_filter: str) -> List[Dict]:
    """Filter news items based on various criteria"""
    
    filtered = news_items
    
    # Category filter
    if category_filter != "All":
        category_key = NEWS_CATEGORIES.get(category_filter, "all")
        if category_key != "all":
            filtered = [item for item in filtered if item.get('category') == category_key]
    
    # Symbol filter
    if symbol_filter:
        symbols = [s.strip().upper() for s in symbol_filter.split(',')]
        filtered = [item for item in filtered if any(
            symbol in (item.get('title', '') + ' ' + item.get('summary', '')).upper() 
            for symbol in symbols
        )]
    
    # Keyword filter
    if keyword_filter:
        keywords = [k.strip().lower() for k in keyword_filter.split(',')]
        filtered = [item for item in filtered if any(
            keyword in (item.get('title', '') + ' ' + item.get('summary', '')).lower()
            for keyword in keywords
        )]
    
    # Sentiment filter
    if sentiment_filter != "All":
        filtered = [item for item in filtered 
                   if item.get('sentiment', {}).get('label', 'Neutral') == sentiment_filter]
    
    return filtered

def search_news_by_criteria(keywords: str, date_from: datetime.date, 
                           date_to: datetime.date, sources: List[str]) -> List[Dict]:
    """Search news by specific criteria"""
    
    all_news = []
    
    for source in sources:
        try:
            news_items = fetch_enhanced_news(source)
            for item in news_items:
                item['source'] = source
                
                # Date filtering
                item_date = item.get('published_parsed', datetime.min).date()
                if date_from <= item_date <= date_to:
                    
                    # Keyword filtering
                    text_content = (item.get('title', '') + ' ' + item.get('summary', '')).lower()
                    if any(keyword.lower() in text_content for keyword in keywords.split()):
                        all_news.append(item)
        except:
            continue
    
    return all_news

def display_news_items(news_items: List[Dict], show_sentiment: bool = True):
    """Display formatted news items with enhanced features"""
    
    st.subheader(f"📰 Latest News ({len(news_items)} articles)")
    
    for i, item in enumerate(news_items[:30]):  # Limit to 30 items for performance
        with st.container():
            # Create expandable news item
            with st.expander(f"**{item.get('title', 'No Title')}**", expanded=(i < 5)):
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # News content
                    if item.get('summary'):
                        summary_text = item['summary']
                        if item.get('summary_generated'):
                            st.markdown(f"*📝 Auto-generated summary:*")
                        st.markdown(f"*{summary_text}*")
                    
                    # Read more link
                    if item.get('link'):
                        st.markdown(f"[🔗 Read full article]({item['link']})")
                    
                    # Show category and source
                    col_cat, col_src = st.columns(2)
                    with col_cat:
                        category = item.get('category', 'general').title()
                        st.markdown(f"**Category:** {category}")
                    with col_src:
                        st.markdown(f"**Source:** {item.get('source', 'Unknown')}")
                
                with col2:
                    # Timestamp
                    if item.get('published_parsed'):
                        time_diff = get_relative_time(item['published_parsed'])
                        st.markdown(f"**📅 {time_diff}**")
                    
                    # Sentiment analysis
                    if show_sentiment and item.get('sentiment'):
                        sentiment = item['sentiment']
                        sentiment_emoji = {
                            'Positive': '😊',
                            'Negative': '😔',
                            'Neutral': '😐'
                        }
                        
                        emoji = sentiment_emoji.get(sentiment['label'], '😐')
                        confidence_pct = sentiment['confidence'] * 100
                        
                        st.markdown(f"**Sentiment:** {emoji} {sentiment['label']}")
                        st.progress(confidence_pct / 100)
                        st.caption(f"Confidence: {confidence_pct:.1f}%")

def get_relative_time(published_date: datetime) -> str:
    """Get relative time string (e.g., '2 hours ago')"""
    try:
        now = datetime.now()
        if published_date.tzinfo is None:
            published_date = published_date.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        
        diff = now - published_date
        
        if diff.days > 1:
            return f"{diff.days} days ago"
        elif diff.days == 1:
            return "1 day ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    except:
        return "Unknown"

def display_news_alerts():
    """Display and manage news alerts system"""
    
    st.subheader("🚨 News Alerts & Notifications")
    
    if not user_auth.is_logged_in():
        st.info("Please login to set up news alerts.")
        return
    
    user_id = user_auth.get_current_user_id()
    
    # Get existing news alerts
    news_alerts = user_auth.get_user_preference("news", "alerts", [])
    
    with st.expander("📢 Configure News Alerts"):
        col1, col2 = st.columns(2)
        
        with col1:
            alert_type = st.selectbox("Alert Type", [
                "Breaking News", 
                "Stock Symbol Mentions", 
                "Earnings Announcements",
                "Merger & Acquisitions",
                "Market Moving Events"
            ])
            
            alert_keywords = st.text_input(
                "Keywords/Symbols", 
                placeholder="e.g., AAPL, Tesla, Federal Reserve"
            )
        
        with col2:
            notification_methods = st.multiselect("Notification Methods", [
                "In-App Notification",
                "Browser Alert",
                "Email (Future)"
            ], default=["In-App Notification"])
            
            alert_frequency = st.selectbox("Check Frequency", [
                "Real-time", "Every 5 minutes", "Every 15 minutes", "Hourly"
            ])
        
        if st.button("➕ Add News Alert"):
            if alert_keywords:
                new_alert = {
                    "id": len(news_alerts) + 1,
                    "type": alert_type,
                    "keywords": alert_keywords,
                    "methods": notification_methods,
                    "frequency": alert_frequency,
                    "created": datetime.now().isoformat(),
                    "active": True,
                    "last_triggered": None
                }
                
                news_alerts.append(new_alert)
                user_auth.auto_save_preference("news", "alerts", news_alerts)
                st.success("News alert added successfully!")
                st.rerun()
    
    # Display existing alerts
    if news_alerts:
        st.subheader("Active News Alerts")
        
        for alert in news_alerts:
            if alert.get('active', True):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{alert['type']}** - Keywords: _{alert['keywords']}_")
                        st.caption(f"Created: {alert.get('created', 'Unknown')} | Frequency: {alert.get('frequency', 'Unknown')}")
                    
                    with col2:
                        status = "🟢 Active" if alert.get('active') else "🔴 Inactive"
                        st.markdown(status)
                    
                    with col3:
                        if st.button(f"🗑️ Remove", key=f"remove_alert_{alert['id']}"):
                            news_alerts = [a for a in news_alerts if a['id'] != alert['id']]
                            user_auth.auto_save_preference("news", "alerts", news_alerts)
                            st.success("Alert removed!")
                            st.rerun()
                    
                    st.markdown("---")
    
    # Check for triggered alerts (simplified implementation)
    check_news_alerts(news_alerts)

def check_news_alerts(alerts: List[Dict]):
    """Check if any news alerts should be triggered"""
    
    if not alerts:
        return
    
    # This is a simplified implementation
    # In a production system, this would run in the background
    try:
        recent_news = []
        for source in list(NEWS_SOURCES.keys())[:3]:  # Check top 3 sources
            try:
                news_items = fetch_enhanced_news(source)
                if news_items:
                    recent_news.extend(news_items[:5])  # Get recent 5 from each
            except:
                continue
        
        for alert in alerts:
            if not alert.get('active'):
                continue
            
            keywords = [k.strip().lower() for k in alert.get('keywords', '').split(',')]
            
            for news_item in recent_news:
                news_text = (news_item.get('title', '') + ' ' + news_item.get('summary', '')).lower()
                
                if any(keyword in news_text for keyword in keywords):
                    # Show notification
                    st.info(f"🚨 **News Alert Triggered**: {alert['type']}")
                    st.markdown(f"**{news_item.get('title', 'News Item')}**")
                    st.markdown(f"_{news_item.get('summary', '')[:150]}..._")
                    break
    
    except Exception as e:
        # Silently handle alert checking errors
        pass

def display_enhanced_market_sentiment():
    """Enhanced market sentiment analysis with multiple indicators"""
    
    st.subheader("📊 Enhanced Market Sentiment Analysis")
    
    # Traditional sentiment indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # VIX Fear & Greed (enhanced)
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="5d")
            
            if not vix_data.empty:
                vix_value = vix_data['Close'].iloc[-1]
                vix_change = vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[-5]
                
                if vix_value > 30:
                    sentiment = "😱 Extreme Fear"
                    color = "#FF0000"
                elif vix_value > 20:
                    sentiment = "😨 High Fear"
                    color = "#FF6600"
                elif vix_value > 15:
                    sentiment = "😐 Moderate Fear"
                    color = "#FFAA00"
                else:
                    sentiment = "😎 Low Fear (Greed)"
                    color = "#00FF00"
                
                st.metric("VIX (Fear Index)", f"{vix_value:.2f}", f"{vix_change:+.2f}")
                st.markdown(f"<div style='color: {color}'>{sentiment}</div>", unsafe_allow_html=True)
        except:
            st.metric("VIX (Fear Index)", "N/A")
    
    with col2:
        # Crypto sentiment
        try:
            btc = yf.Ticker("BTC-USD")
            btc_data = btc.history(period="5d")
            
            if not btc_data.empty:
                btc_change = ((btc_data['Close'].iloc[-1] - btc_data['Close'].iloc[-5]) / btc_data['Close'].iloc[-5]) * 100
                
                if btc_change > 5:
                    crypto_sentiment = "🚀 Crypto Bullish"
                elif btc_change < -5:
                    crypto_sentiment = "📉 Crypto Bearish"
                else:
                    crypto_sentiment = "➡️ Crypto Neutral"
                
                st.metric("Crypto Sentiment", f"${btc_data['Close'].iloc[-1]:,.0f}", f"{btc_change:+.1f}%")
                st.markdown(crypto_sentiment)
        except:
            st.metric("Crypto Sentiment", "N/A")
    
    with col3:
        # Commodity sentiment (Gold)
        try:
            gold = yf.Ticker("GC=F")
            gold_data = gold.history(period="5d")
            
            if not gold_data.empty:
                gold_change = ((gold_data['Close'].iloc[-1] - gold_data['Close'].iloc[-5]) / gold_data['Close'].iloc[-5]) * 100
                
                if gold_change > 2:
                    commodity_sentiment = "🥇 Safe Haven Demand"
                elif gold_change < -2:
                    commodity_sentiment = "📉 Risk-On Mode"
                else:
                    commodity_sentiment = "➡️ Stable Commodities"
                
                st.metric("Gold Sentiment", f"${gold_data['Close'].iloc[-1]:,.0f}", f"{gold_change:+.1f}%")
                st.markdown(commodity_sentiment)
        except:
            st.metric("Gold Sentiment", "N/A")
    
    with col4:
        # Dollar strength
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_data = dxy.history(period="5d")
            
            if not dxy_data.empty:
                dxy_change = ((dxy_data['Close'].iloc[-1] - dxy_data['Close'].iloc[-5]) / dxy_data['Close'].iloc[-5]) * 100
                
                if dxy_change > 1:
                    dollar_sentiment = "💪 Strong Dollar"
                elif dxy_change < -1:
                    dollar_sentiment = "📉 Weak Dollar"
                else:
                    dollar_sentiment = "➡️ Stable Dollar"
                
                st.metric("Dollar Index", f"{dxy_data['Close'].iloc[-1]:.2f}", f"{dxy_change:+.2f}%")
                st.markdown(dollar_sentiment)
        except:
            st.metric("Dollar Index", "N/A")
    
    # Market heat map visualization
    display_market_heatmap()

def display_market_heatmap():
    """Display market heat map visualization"""
    
    st.subheader("🗺️ Market Heat Map")
    
    try:
        # Get data for major indices and sectors
        symbols = ["^GSPC", "^IXIC", "^DJI", "^RUT", "XLK", "XLF", "XLE", "XLV", "XLI", "XLC"]
        names = ["S&P 500", "NASDAQ", "DOW", "Russell 2000", "Technology", "Financial", "Energy", "Healthcare", "Industrial", "Communication"]
        
        heatmap_data = []
        
        for symbol, name in zip(symbols, names):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    change_pct = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                    heatmap_data.append({
                        'Symbol': symbol,
                        'Name': name,
                        'Change %': change_pct,
                        'Color': 'green' if change_pct >= 0 else 'red'
                    })
            except:
                continue
        
        if heatmap_data:
            df = pd.DataFrame(heatmap_data)
            
            # Create color-coded metrics
            cols = st.columns(min(5, len(heatmap_data)))
            for i, row in df.iterrows():
                col_idx = i % 5
                with cols[col_idx]:
                    color = "#00AA00" if row['Change %'] >= 0 else "#AA0000"
                    st.markdown(f"**{row['Name']}**")
                    st.markdown(f"<div style='color: {color}; font-weight: bold'>{row['Change %']:+.2f}%</div>", 
                               unsafe_allow_html=True)
    except:
        st.info("Market heat map temporarily unavailable")

def display_fallback_news():
    """Enhanced fallback news when RSS feeds fail"""
    st.info("📰 Live news feeds are temporarily unavailable. Here are key market areas to monitor:")
    
    fallback_topics = [
        "📈 **Federal Reserve Policy** - FOMC meetings and interest rate decisions affect all markets",
        "📊 **Earnings Season** - Quarterly reports from major corporations drive individual stock performance", 
        "🌍 **Geopolitical Events** - Global tensions, trade wars, and political developments impact market volatility",
        "💱 **Currency Markets** - USD strength affects international trade, emerging markets, and commodity prices",
        "🛢️ **Energy & Commodities** - Oil prices, gold movements, and agricultural commodity trends affect inflation",
        "🏢 **Economic Data** - GDP growth, employment numbers, inflation reports drive market sentiment",
        "💻 **Technology Sector** - AI developments, regulatory changes, and innovation cycles affect growth stocks",
        "🏦 **Financial System** - Banking sector health, credit conditions, and regulatory changes",
        "🏠 **Real Estate** - Housing market data affects REITs, construction, and consumer spending patterns",
        "🌱 **ESG & Climate** - Environmental regulations and sustainability trends increasingly impact valuations",
        "₿ **Cryptocurrency** - Digital asset adoption, regulation, and institutional investment trends",
        "🔬 **Biotech & Healthcare** - Drug approvals, clinical trials, and healthcare policy developments"
    ]
    
    for topic in fallback_topics:
        st.markdown(topic)
        st.markdown("")
    
    # Add market monitoring tips
    st.subheader("💡 Market Monitoring Tips")
    st.markdown("""
    **Key Times to Watch:**
    - 📅 **Pre-market (4:00-9:30 AM ET)** - Overnight developments and earnings releases
    - 🔔 **Market Open (9:30 AM ET)** - Initial reaction to news and overnight events  
    - 📊 **2:00 PM ET** - FOMC announcements and economic data releases
    - 🌅 **After Hours (4:00-8:00 PM ET)** - Earnings calls and corporate announcements
    - 🌏 **Overnight** - Asian and European market movements affect US pre-market
    
    **Reliable News Sources:**
    - Bloomberg Terminal, Reuters, MarketWatch for breaking financial news
    - Company investor relations pages for official announcements
    - Federal Reserve and Treasury websites for policy updates
    - SEC filings (10-K, 10-Q, 8-K) for detailed company information
    """)

# Mark task as completed and move to next one