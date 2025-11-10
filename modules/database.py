import os
import psycopg2
import psycopg2.extras
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import streamlit as st

class DatabaseManager:
    """Database manager for user profile system with auto-save functionality"""
    
    def __init__(self):
        self.connection = None
        self.database_url = os.getenv('DATABASE_URL')
        
    def get_connection(self):
        """Get database connection"""
        if self.connection is None or self.connection.closed:
            try:
                self.connection = psycopg2.connect(self.database_url)
                self.connection.autocommit = True
            except Exception as e:
                st.error(f"Database connection error: {str(e)}")
                return None
        return self.connection
    
    def close_connection(self):
        """Close database connection"""
        if self.connection and not self.connection.closed:
            self.connection.close()
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict]]:
        """Execute a query and return results"""
        try:
            conn = self.get_connection()
            if conn is None:
                return None
                
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
        except Exception as e:
            st.error(f"Database query error: {str(e)}")
            return None
    
    def execute_single(self, query: str, params: tuple = None) -> Optional[Dict]:
        """Execute a query and return single result"""
        results = self.execute_query(query, params)
        return results[0] if results else None
    
    # User Management
    def create_user(self, username: str) -> Optional[Dict]:
        """Create a new user"""
        query = """
        INSERT INTO users (username) 
        VALUES (%s) 
        ON CONFLICT (username) DO NOTHING
        RETURNING id, username, created_at
        """
        return self.execute_single(query, (username,))
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        query = "SELECT * FROM users WHERE username = %s AND is_active = TRUE"
        return self.execute_single(query, (username,))
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        query = "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s"
        self.execute_query(query, (user_id,))
    
    # Portfolio Management
    def save_portfolio_position(self, user_id: int, symbol: str, shares: float, 
                              purchase_price: float, purchase_date: date) -> bool:
        """Save or update a portfolio position"""
        try:
            query = """
            INSERT INTO user_portfolios (user_id, symbol, shares, purchase_price, purchase_date)
            VALUES (%s, %s, %s, %s, %s)
            """
            self.execute_query(query, (user_id, symbol.upper(), shares, purchase_price, purchase_date))
            return True
        except Exception as e:
            st.error(f"Error saving portfolio position: {str(e)}")
            return False
    
    def get_user_portfolio(self, user_id: int) -> List[Dict]:
        """Get all portfolio positions for a user"""
        query = """
        SELECT * FROM user_portfolios 
        WHERE user_id = %s 
        ORDER BY created_at DESC
        """
        results = self.execute_query(query, (user_id,))
        return results or []
    
    def remove_portfolio_position(self, user_id: int, position_id: int) -> bool:
        """Remove a portfolio position"""
        try:
            query = "DELETE FROM user_portfolios WHERE id = %s AND user_id = %s"
            self.execute_query(query, (position_id, user_id))
            return True
        except Exception as e:
            st.error(f"Error removing portfolio position: {str(e)}")
            return False
    
    def clear_user_portfolio(self, user_id: int) -> bool:
        """Clear all portfolio positions for a user"""
        try:
            query = "DELETE FROM user_portfolios WHERE user_id = %s"
            self.execute_query(query, (user_id,))
            return True
        except Exception as e:
            st.error(f"Error clearing portfolio: {str(e)}")
            return False
    
    # Alert Management
    def save_alert(self, user_id: int, alert_data: Dict) -> bool:
        """Save an alert"""
        try:
            query = """
            INSERT INTO user_alerts (
                user_id, alert_type, symbol, alert_name, condition_type,
                target_price, lower_price, upper_price, target_volume,
                notification_methods, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                user_id,
                alert_data.get('alert_type'),
                alert_data.get('symbol', '').upper(),
                alert_data.get('alert_name'),
                alert_data.get('condition'),
                alert_data.get('target_price'),
                alert_data.get('lower_price'),
                alert_data.get('upper_price'),
                alert_data.get('target_volume'),
                json.dumps(alert_data.get('notification_method', [])),
                alert_data.get('status', 'Active')
            )
            self.execute_query(query, params)
            return True
        except Exception as e:
            st.error(f"Error saving alert: {str(e)}")
            return False
    
    def get_user_alerts(self, user_id: int, status: str = None) -> List[Dict]:
        """Get all alerts for a user"""
        if status:
            query = "SELECT * FROM user_alerts WHERE user_id = %s AND status = %s ORDER BY created_at DESC"
            params = (user_id, status)
        else:
            query = "SELECT * FROM user_alerts WHERE user_id = %s ORDER BY created_at DESC"
            params = (user_id,)
        
        results = self.execute_query(query, params)
        
        # Parse JSON fields
        if results:
            for alert in results:
                if alert.get('notification_methods'):
                    try:
                        alert['notification_methods'] = json.loads(alert['notification_methods'])
                    except:
                        alert['notification_methods'] = []
        
        return results or []
    
    def update_alert_status(self, user_id: int, alert_id: int, status: str) -> bool:
        """Update alert status"""
        try:
            query = "UPDATE user_alerts SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s AND user_id = %s"
            self.execute_query(query, (status, alert_id, user_id))
            return True
        except Exception as e:
            st.error(f"Error updating alert status: {str(e)}")
            return False
    
    def remove_alert(self, user_id: int, alert_id: int) -> bool:
        """Remove an alert"""
        try:
            query = "DELETE FROM user_alerts WHERE id = %s AND user_id = %s"
            self.execute_query(query, (alert_id, user_id))
            return True
        except Exception as e:
            st.error(f"Error removing alert: {str(e)}")
            return False
    
    def clear_user_alerts(self, user_id: int) -> bool:
        """Clear all alerts for a user"""
        try:
            query = "DELETE FROM user_alerts WHERE user_id = %s"
            self.execute_query(query, (user_id,))
            return True
        except Exception as e:
            st.error(f"Error clearing alerts: {str(e)}")
            return False
    
    # Watchlist Management
    def add_to_watchlist(self, user_id: int, symbol: str) -> bool:
        """Add symbol to user's watchlist"""
        try:
            query = """
            INSERT INTO user_watchlists (user_id, symbol)
            VALUES (%s, %s)
            ON CONFLICT (user_id, symbol) DO NOTHING
            """
            self.execute_query(query, (user_id, symbol.upper()))
            return True
        except Exception as e:
            st.error(f"Error adding to watchlist: {str(e)}")
            return False
    
    def remove_from_watchlist(self, user_id: int, symbol: str) -> bool:
        """Remove symbol from user's watchlist"""
        try:
            query = "DELETE FROM user_watchlists WHERE user_id = %s AND symbol = %s"
            self.execute_query(query, (user_id, symbol.upper()))
            return True
        except Exception as e:
            st.error(f"Error removing from watchlist: {str(e)}")
            return False
    
    def get_user_watchlist(self, user_id: int) -> List[str]:
        """Get user's watchlist symbols"""
        query = "SELECT symbol FROM user_watchlists WHERE user_id = %s ORDER BY added_at DESC"
        results = self.execute_query(query, (user_id,))
        return [row['symbol'] for row in results] if results else []
    
    def save_watchlist(self, user_id: int, symbols: List[str]) -> bool:
        """Save entire watchlist (replace existing)"""
        try:
            # Clear existing watchlist
            self.execute_query("DELETE FROM user_watchlists WHERE user_id = %s", (user_id,))
            
            # Add new symbols
            for symbol in symbols:
                if symbol.strip():
                    self.add_to_watchlist(user_id, symbol.strip())
            return True
        except Exception as e:
            st.error(f"Error saving watchlist: {str(e)}")
            return False
    
    # Preferences Management
    def save_preference(self, user_id: int, preference_type: str, 
                       preference_key: str, preference_value: Any) -> bool:
        """Save a user preference"""
        try:
            query = """
            INSERT INTO user_preferences (user_id, preference_type, preference_key, preference_value)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, preference_type, preference_key) 
            DO UPDATE SET preference_value = EXCLUDED.preference_value, updated_at = CURRENT_TIMESTAMP
            """
            self.execute_query(query, (user_id, preference_type, preference_key, json.dumps(preference_value)))
            return True
        except Exception as e:
            st.error(f"Error saving preference: {str(e)}")
            return False
    
    def get_preference(self, user_id: int, preference_type: str, preference_key: str) -> Any:
        """Get a specific user preference"""
        query = """
        SELECT preference_value FROM user_preferences 
        WHERE user_id = %s AND preference_type = %s AND preference_key = %s
        """
        result = self.execute_single(query, (user_id, preference_type, preference_key))
        if result and result.get('preference_value'):
            try:
                return json.loads(result['preference_value'])
            except:
                return result['preference_value']
        return None
    
    def get_user_preferences(self, user_id: int, preference_type: str = None) -> Dict:
        """Get all preferences for a user of a specific type"""
        if preference_type:
            query = """
            SELECT preference_key, preference_value FROM user_preferences 
            WHERE user_id = %s AND preference_type = %s
            """
            params = (user_id, preference_type)
        else:
            query = """
            SELECT preference_type, preference_key, preference_value FROM user_preferences 
            WHERE user_id = %s
            """
            params = (user_id,)
        
        results = self.execute_query(query, params)
        preferences = {}
        
        if results:
            for row in results:
                try:
                    value = json.loads(row['preference_value'])
                except:
                    value = row['preference_value']
                
                if preference_type:
                    preferences[row['preference_key']] = value
                else:
                    pref_type = row['preference_type']
                    if pref_type not in preferences:
                        preferences[pref_type] = {}
                    preferences[pref_type][row['preference_key']] = value
        
        return preferences
    
    def remove_preference(self, user_id: int, preference_type: str, preference_key: str) -> bool:
        """Remove a specific preference"""
        try:
            query = """
            DELETE FROM user_preferences 
            WHERE user_id = %s AND preference_type = %s AND preference_key = %s
            """
            self.execute_query(query, (user_id, preference_type, preference_key))
            return True
        except Exception as e:
            st.error(f"Error removing preference: {str(e)}")
            return False

# Global database manager instance
db_manager = DatabaseManager()