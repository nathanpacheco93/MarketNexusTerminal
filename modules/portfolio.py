import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from modules.user_auth import user_auth

def display_portfolio():
    """Display portfolio management section"""
    
    # Portfolio input section
    st.subheader("➕ Add Position")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL")
    
    with col2:
        shares = st.number_input("Shares", min_value=0.0, step=0.1)
    
    with col3:
        purchase_price = st.number_input("Purchase Price ($)", min_value=0.0, step=0.01)
    
    with col4:
        purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
    
    if st.button("Add Position"):
        if symbol and shares > 0 and purchase_price > 0:
            new_position = {
                'symbol': symbol.upper(),
                'shares': shares,
                'purchase_price': purchase_price,
                'purchase_date': purchase_date,
                'purchase_value': shares * purchase_price
            }
            
            # Auto-save to database
            if user_auth.auto_save_portfolio_position(symbol.upper(), shares, purchase_price, purchase_date):
                st.session_state.portfolio.append(new_position)
                st.success(f"Added {shares} shares of {symbol.upper()} to portfolio")
                st.rerun()
            else:
                st.error("Failed to save position to database")
        else:
            st.error("Please fill in all fields with valid values")
    
    # Display current portfolio
    if st.session_state.portfolio:
        st.subheader("💼 Current Portfolio")
        
        # Calculate portfolio metrics
        portfolio_data = []
        total_invested = 0
        total_current_value = 0
        
        for position in st.session_state.portfolio:
            try:
                # Get current price
                ticker = yf.Ticker(position['symbol'])
                current_data = ticker.history(period="1d")
                
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    current_value = position['shares'] * current_price
                    gain_loss = current_value - position['purchase_value']
                    gain_loss_percent = (gain_loss / position['purchase_value']) * 100
                    
                    portfolio_data.append({
                        'Symbol': position['symbol'],
                        'Shares': position['shares'],
                        'Purchase Price': f"${position['purchase_price']:.2f}",
                        'Current Price': f"${current_price:.2f}",
                        'Purchase Value': f"${position['purchase_value']:.2f}",
                        'Current Value': f"${current_value:.2f}",
                        'Gain/Loss': f"${gain_loss:.2f}",
                        'Gain/Loss %': f"{gain_loss_percent:.2f}%",
                        'Purchase Date': position['purchase_date'].strftime("%Y-%m-%d")
                    })
                    
                    total_invested += position['purchase_value']
                    total_current_value += current_value
                    
            except Exception as e:
                st.error(f"Error fetching data for {position['symbol']}: {str(e)}")
        
        if portfolio_data:
            # Portfolio summary metrics
            total_gain_loss = total_current_value - total_invested
            total_gain_loss_percent = (total_gain_loss / total_invested) * 100 if total_invested > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Invested", f"${total_invested:.2f}")
            
            with col2:
                st.metric("Current Value", f"${total_current_value:.2f}")
            
            with col3:
                st.metric(
                    "Total Gain/Loss", 
                    f"${total_gain_loss:.2f}",
                    delta=f"{total_gain_loss_percent:.2f}%"
                )
            
            with col4:
                performance = "📈 Positive" if total_gain_loss >= 0 else "📉 Negative"
                st.metric("Performance", performance)
            
            # Portfolio allocation pie chart
            if len(portfolio_data) > 1:
                st.subheader("📊 Portfolio Allocation")
                
                # Create allocation data
                allocation_data = []
                for item in portfolio_data:
                    allocation_data.append({
                        'Symbol': item['Symbol'],
                        'Value': float(item['Current Value'].replace('$', '').replace(',', ''))
                    })
                
                df_allocation = pd.DataFrame(allocation_data)
                
                fig = px.pie(
                    df_allocation, 
                    values='Value', 
                    names='Symbol',
                    title="Portfolio Allocation by Current Value"
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio performance chart
            st.subheader("📈 Portfolio Performance")
            
            # Create performance data for chart
            performance_data = []
            for item in portfolio_data:
                symbol = item['Symbol']
                gain_loss_str = item['Gain/Loss %'].replace('%', '')
                gain_loss_percent = float(gain_loss_str)
                
                performance_data.append({
                    'Symbol': symbol,
                    'Performance': gain_loss_percent
                })
            
            df_performance = pd.DataFrame(performance_data)
            
            # Color coding for performance
            colors = ['green' if x >= 0 else 'red' for x in df_performance['Performance']]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df_performance['Symbol'],
                    y=df_performance['Performance'],
                    marker_color=colors,
                    text=[f"{x:.1f}%" for x in df_performance['Performance']],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Individual Stock Performance (%)",
                xaxis_title="Symbol",
                yaxis_title="Performance (%)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio table
            st.subheader("📋 Position Details")
            df = pd.DataFrame(portfolio_data)
            st.dataframe(df, use_container_width=True)
            
            # Remove position functionality
            st.subheader("🗑️ Remove Positions")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                symbols_to_remove = [pos['symbol'] for pos in st.session_state.portfolio]
                symbol_to_remove = st.selectbox("Select position to remove", symbols_to_remove)
            
            with col2:
                if st.button("Remove Position"):
                    if symbol_to_remove:
                        # Find and remove the position
                        for i, position in enumerate(st.session_state.portfolio):
                            if position['symbol'] == symbol_to_remove:
                                # Try to remove from database first
                                from modules.database import db_manager
                                user_id = user_auth.get_current_user_id()
                                
                                if user_id and position.get('id'):
                                    # Remove from database using position ID
                                    if db_manager.remove_portfolio_position(user_id, position['id']):
                                        removed_position = st.session_state.portfolio.pop(i)
                                        st.success(f"Removed {removed_position['shares']} shares of {symbol_to_remove}")
                                        st.rerun()
                                        break
                                    else:
                                        st.error("Failed to remove position from database")
                                else:
                                    # Fallback for positions without ID (legacy)
                                    removed_position = st.session_state.portfolio.pop(i)
                                    st.success(f"Removed {removed_position['shares']} shares of {symbol_to_remove}")
                                    st.rerun()
                                    break
        
        # Export portfolio functionality
        if st.button("📊 Export Portfolio Data"):
            if portfolio_data:
                df = pd.DataFrame(portfolio_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio CSV",
                    data=csv,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("Your portfolio is empty. Add some positions to get started!")
        
        # Sample portfolio suggestions
        st.subheader("💡 Sample Portfolio Ideas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Technology Portfolio:**
            - AAPL (Apple)
            - GOOGL (Alphabet)
            - MSFT (Microsoft)
            - NVDA (NVIDIA)
            - TSLA (Tesla)
            """)
        
        with col2:
            st.markdown("""
            **Diversified Portfolio:**
            - SPY (S&P 500 ETF)
            - VTI (Total Stock Market)
            - BND (Bond ETF)
            - GLD (Gold ETF)
            - VEA (International ETF)
            """)

def calculate_portfolio_beta(portfolio_data, benchmark_symbol="SPY"):
    """Calculate portfolio beta relative to benchmark"""
    try:
        # This would require more complex calculations
        # For now, return a placeholder
        return 1.0
    except:
        return None
