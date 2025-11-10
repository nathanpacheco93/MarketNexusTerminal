# Bloomberg Terminal Clone

## Overview

This is a comprehensive financial data terminal application built with Streamlit that replicates core Bloomberg Terminal functionality. The application provides real-time market data, interactive charts, options pricing, portfolio management, risk analytics, and various financial tools for traders and analysts.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Wide layout with expandable sidebar navigation
- **Styling**: Custom CSS for Bloomberg-inspired dark theme with metric containers
- **Visualization**: Plotly for interactive charts and financial data visualization
- **Navigation**: Tab-based interface within modules for organized feature access

### Application Structure
- **Modular Design**: Each major feature is separated into dedicated modules in the `/modules` directory
- **Main Application**: `app.py` serves as the entry point with navigation and page routing
- **Module System**: Eleven specialized modules handle different financial functions:
  - Market data and overview
  - Interactive charting
  - Options chain and derivatives
  - Strategy backtesting
  - Risk management and portfolio analytics
  - Bond market analysis
  - Alert and notification system
  - Portfolio management
  - Market screening
  - Financial news aggregation
  - Economic calendar

### Data Management
- **Data Source**: Yahoo Finance (yfinance) as primary data provider
- **Caching Strategy**: Streamlit's `@st.cache_data` with 60-second TTL for real-time data
- **User Profile System**: Comprehensive PostgreSQL database for persistent user data storage
- **Session State**: Used for maintaining portfolio data, alerts, and user preferences across page refreshes
- **Data Processing**: Pandas for data manipulation and NumPy for financial calculations
- **Auto-Save**: Automatic persistence of user data across all modules

### Financial Analytics Engine
- **Options Pricing**: Black-Scholes model implementation using SciPy
- **Risk Calculations**: VaR, volatility, correlation analysis, and portfolio optimization
- **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands
- **Backtesting Framework**: Historical strategy testing with performance metrics
- **Bond Analytics**: Yield curve analysis and fixed income calculations

### Real-time Features
- **Market Data**: Live price feeds for stocks, indices, currencies, and commodities
- **Alert System**: Price-based and technical indicator alerts with notification management
- **Auto-refresh**: Configurable automatic data updates for market monitoring
- **User Profile System**: Real-time auto-save of user preferences, portfolio, alerts, and watchlists

### User Profile & Persistence System
- **User Authentication**: Simple username-based identification system
- **Database Schema**: PostgreSQL tables for users, portfolios, alerts, watchlists, and preferences
- **Auto-Save Functionality**: Seamless persistence across all modules:
  - Portfolio holdings and transactions
  - Active alerts and alert history
  - Watchlist symbols and favorites
  - Chart preferences and indicator settings
  - Personal user settings and configurations
- **Profile Management**: Comprehensive interface for data overview, export, and management
- **Data Integrity**: Robust CRUD operations with error handling and validation

## External Dependencies

### Data Providers
- **Yahoo Finance API**: Primary source for stock prices, options data, and market information via `yfinance` library
- **RSS Feeds**: Financial news from Yahoo Finance, MarketWatch, Reuters, Bloomberg, and Financial Times
- **Economic Calendar**: Simulated economic events (designed for future integration with real APIs)

### Python Libraries
- **Core Framework**: `streamlit` for web application interface
- **Data Processing**: `pandas` for data manipulation, `numpy` for numerical computations
- **Visualization**: `plotly` for interactive charts and financial plots
- **Financial Calculations**: `scipy` for optimization and statistical functions
- **Web Requests**: `requests` for API calls and data fetching
- **Feed Parsing**: `feedparser` for RSS news feed processing

### Mathematical Models
- **Options Pricing**: Black-Scholes model for derivatives valuation
- **Risk Models**: Modern Portfolio Theory, VaR calculations, correlation analysis
- **Technical Indicators**: Standard technical analysis formulas and momentum indicators
- **Bond Pricing**: Fixed income analytics and yield curve interpolation

### Recent Changes (September 2025)
- **User Profile System**: Implemented comprehensive user authentication and data persistence
- **Database Integration**: Added PostgreSQL database with 5 tables for user data management
- **Auto-Save Functionality**: Integrated throughout all modules for seamless data persistence
- **Profile Management**: Added dedicated interface for user data management and export
- **Enhanced User Experience**: Personalized settings and preferences that persist across sessions

### Database Architecture
- **users**: User authentication and account information
- **user_portfolios**: Portfolio holdings, transactions, and position tracking
- **user_alerts**: Alert configurations, triggers, and notification history
- **user_watchlists**: Personal symbol watchlists and favorites
- **user_preferences**: Chart settings, indicator preferences, and custom configurations

### Limitations and Considerations
- Real-time data limited by Yahoo Finance API rate limits and availability
- No direct Bloomberg Terminal or professional market data integration
- Economic calendar uses simulated data pending API integration
- User data persists in development database (production deployment may require migration)