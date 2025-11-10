import streamlit as st
from typing import Optional, Dict
from modules.database import db_manager

class UserAuth:
    """Simple username-based user authentication and identification system"""
    
    @staticmethod
    def initialize_session():
        """Initialize user session state variables"""
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_data' not in st.session_state:
            st.session_state.user_data = None
        if 'auto_save_enabled' not in st.session_state:
            st.session_state.auto_save_enabled = True
    
    @staticmethod
    def get_current_user() -> Optional[Dict]:
        """Get current logged-in user data"""
        UserAuth.initialize_session()
        if st.session_state.user_id and st.session_state.user_data:
            return st.session_state.user_data
        return None
    
    @staticmethod
    def get_current_user_id() -> Optional[int]:
        """Get current user ID"""
        UserAuth.initialize_session()
        return st.session_state.user_id
    
    @staticmethod
    def is_logged_in() -> bool:
        """Check if user is logged in"""
        UserAuth.initialize_session()
        return st.session_state.user_id is not None
    
    @staticmethod
    def login_user(username: str) -> bool:
        """Login user with username (create if doesn't exist)"""
        if not username or not username.strip():
            st.error("Please enter a valid username")
            return False
        
        username = username.strip().lower()
        
        try:
            # Try to get existing user
            user = db_manager.get_user(username)
            
            if user:
                # Existing user
                st.session_state.user_id = user['id']
                st.session_state.username = user['username']
                st.session_state.user_data = user
                
                # Update last login
                db_manager.update_last_login(user['id'])
                
                st.success(f"Welcome back, {username}!")
            else:
                # Create new user
                user = db_manager.create_user(username)
                if user:
                    st.session_state.user_id = user['id']
                    st.session_state.username = user['username']
                    st.session_state.user_data = user
                    
                    st.success(f"Welcome, {username}! Your profile has been created.")
                else:
                    st.error("Failed to create user profile")
                    return False
            
            # Load user data after login
            UserAuth.load_user_data()
            return True
            
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
            return False
    
    @staticmethod
    def logout_user():
        """Logout current user"""
        # Save current state before logout
        if UserAuth.is_logged_in():
            UserAuth.save_current_state()
        
        # Clear session state
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.user_data = None
        
        # Clear other session data
        for key in ['portfolio', 'alerts', 'alert_history', 'watchlist']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("Logged out successfully")
        st.rerun()
    
    @staticmethod
    def load_user_data():
        """Load user data from database into session state"""
        user_id = UserAuth.get_current_user_id()
        if not user_id:
            return
        
        try:
            # Load portfolio
            portfolio_data = db_manager.get_user_portfolio(user_id)
            portfolio_list = []
            for pos in portfolio_data:
                portfolio_list.append({
                    'id': pos['id'],
                    'symbol': pos['symbol'],
                    'shares': float(pos['shares']),
                    'purchase_price': float(pos['purchase_price']),
                    'purchase_date': pos['purchase_date'],
                    'purchase_value': float(pos['purchase_value'])
                })
            st.session_state.portfolio = portfolio_list
            
            # Load alerts
            alerts_data = db_manager.get_user_alerts(user_id, status='Active')
            alerts_list = []
            for alert in alerts_data:
                alert_dict = {
                    'id': alert['id'],
                    'alert_type': alert['alert_type'],
                    'symbol': alert['symbol'],
                    'alert_name': alert['alert_name'],
                    'condition': alert['condition_type'],
                    'notification_method': alert.get('notification_methods', []),
                    'status': alert['status'],
                    'triggered_count': alert['triggered_count'],
                    'created_at': alert['created_at'].isoformat() if alert['created_at'] else None
                }
                
                # Add price/volume conditions
                if alert['target_price']:
                    alert_dict['target_price'] = float(alert['target_price'])
                if alert['lower_price']:
                    alert_dict['lower_price'] = float(alert['lower_price'])
                if alert['upper_price']:
                    alert_dict['upper_price'] = float(alert['upper_price'])
                if alert['target_volume']:
                    alert_dict['target_volume'] = alert['target_volume']
                
                alerts_list.append(alert_dict)
            
            st.session_state.alerts = alerts_list
            
            # Load alert history
            alert_history = db_manager.get_user_alerts(user_id)
            st.session_state.alert_history = alert_history
            
            # Load watchlist
            watchlist = db_manager.get_user_watchlist(user_id)
            st.session_state.watchlist = watchlist if watchlist else ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN']
            
        except Exception as e:
            st.error(f"Error loading user data: {str(e)}")
    
    @staticmethod
    def save_current_state():
        """Save current session state to database"""
        user_id = UserAuth.get_current_user_id()
        if not user_id or not st.session_state.get('auto_save_enabled', True):
            return
        
        try:
            # Save watchlist
            if 'watchlist' in st.session_state:
                db_manager.save_watchlist(user_id, st.session_state.watchlist)
            
        except Exception as e:
            st.error(f"Error saving user data: {str(e)}")
    
    @staticmethod
    def auto_save_portfolio_position(symbol: str, shares: float, purchase_price: float, purchase_date):
        """Auto-save a new portfolio position"""
        user_id = UserAuth.get_current_user_id()
        if not user_id or not st.session_state.get('auto_save_enabled', True):
            return False
        
        return db_manager.save_portfolio_position(user_id, symbol, shares, purchase_price, purchase_date)
    
    @staticmethod
    def auto_save_alert(alert_data: Dict):
        """Auto-save a new alert"""
        user_id = UserAuth.get_current_user_id()
        if not user_id or not st.session_state.get('auto_save_enabled', True):
            return False
        
        return db_manager.save_alert(user_id, alert_data)
    
    @staticmethod
    def auto_save_watchlist_symbol(symbol: str, action: str = 'add'):
        """Auto-save watchlist changes"""
        user_id = UserAuth.get_current_user_id()
        if not user_id or not st.session_state.get('auto_save_enabled', True):
            return False
        
        if action == 'add':
            return db_manager.add_to_watchlist(user_id, symbol)
        elif action == 'remove':
            return db_manager.remove_from_watchlist(user_id, symbol)
        return False
    
    @staticmethod
    def auto_save_preference(preference_type: str, preference_key: str, preference_value):
        """Auto-save user preference"""
        user_id = UserAuth.get_current_user_id()
        if not user_id or not st.session_state.get('auto_save_enabled', True):
            return False
        
        return db_manager.save_preference(user_id, preference_type, preference_key, preference_value)
    
    @staticmethod
    def get_user_preference(preference_type: str, preference_key: str, default_value=None):
        """Get user preference with default value"""
        user_id = UserAuth.get_current_user_id()
        if not user_id:
            return default_value
        
        pref_value = db_manager.get_preference(user_id, preference_type, preference_key)
        return pref_value if pref_value is not None else default_value
    
    @staticmethod
    def display_login_form():
        """Display login form for user authentication"""
        st.markdown("### 🔐 User Login")
        
        with st.form("login_form"):
            st.markdown("Enter your username to access your personalized Bloomberg Terminal:")
            username = st.text_input(
                "Username", 
                placeholder="Enter your username",
                help="Create a new account by entering a new username, or login with an existing one"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                login_btn = st.form_submit_button("🚀 Login / Create Account", use_container_width=True)
            with col2:
                if st.form_submit_button("ℹ️ About Profiles", use_container_width=True):
                    st.info("""
                    **User Profiles provide:**
                    - Persistent portfolio tracking
                    - Saved alerts and notifications
                    - Custom watchlists
                    - Chart preferences and settings
                    - Screener filters and saved searches
                    - Auto-save functionality across all modules
                    """)
            
            if login_btn and username:
                if UserAuth.login_user(username):
                    st.rerun()
        
        # Display some benefits of using profiles
        st.markdown("---")
        st.markdown("### ✨ Benefits of User Profiles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Portfolio Tracking**
            - Persistent holdings
            - Performance history
            - Auto-save positions
            """)
        
        with col2:
            st.markdown("""
            **🚨 Alert Management**
            - Price alerts
            - Volume alerts
            - Technical indicators
            """)
        
        with col3:
            st.markdown("""
            **⚙️ Personal Settings**
            - Chart preferences
            - Saved watchlists
            - Custom screener filters
            """)
    
    @staticmethod
    def display_user_menu():
        """Display user menu in sidebar"""
        if not UserAuth.is_logged_in():
            return
        
        user = UserAuth.get_current_user()
        if user:
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"👤 **Logged in as:** {user['username']}")
            st.sidebar.markdown(f"📅 **Member since:** {user['created_at'].strftime('%Y-%m-%d') if user['created_at'] else 'Unknown'}")
            
            # Auto-save toggle
            st.session_state.auto_save_enabled = st.sidebar.checkbox(
                "💾 Auto-save enabled", 
                value=st.session_state.get('auto_save_enabled', True),
                help="Automatically save portfolio, alerts, and preferences"
            )
            
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("💾 Save Now", help="Manually save current state"):
                    UserAuth.save_current_state()
                    st.success("Data saved!")
            
            with col2:
                if st.button("🚪 Logout"):
                    UserAuth.logout_user()

# Initialize auth system
user_auth = UserAuth()