# modules/user_auth.py — NO-DB / NO-LOGIN VERSION

import streamlit as st
from typing import Optional, Dict

class UserAuth:
    """
    Lightweight session-only 'auth' shim.
    - No database
    - No real login
    - Keeps the same public methods as the original so other modules don't break.
    """

    DEFAULT_USERNAME = "local"
    DEFAULT_USER_ID = 0

    @staticmethod
    def initialize_session():
        """Initialize session keys used across the app."""
        if "user_id" not in st.session_state:
            st.session_state.user_id = UserAuth.DEFAULT_USER_ID
        if "username" not in st.session_state:
            st.session_state.username = UserAuth.DEFAULT_USERNAME
        if "user_data" not in st.session_state:
            st.session_state.user_data = {
                "id": UserAuth.DEFAULT_USER_ID,
                "username": UserAuth.DEFAULT_USERNAME,
                "created_at": None,  # no DB date
            }
        if "auto_save_enabled" not in st.session_state:
            st.session_state.auto_save_enabled = True

        # Local-only caches/state used by pages
        if "portfolio" not in st.session_state:
            st.session_state.portfolio = []
        if "alerts" not in st.session_state:
            st.session_state.alerts = []
        if "alert_history" not in st.session_state:
            st.session_state.alert_history = []
        if "watchlist" not in st.session_state:
            st.session_state.watchlist = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN"]
        if "preferences" not in st.session_state:
            st.session_state.preferences = {}  # {(type, key): value}

    @staticmethod
    def get_current_user() -> Optional[Dict]:
        """Return the pseudo-user dict."""
        UserAuth.initialize_session()
        return st.session_state.user_data

    @staticmethod
    def get_current_user_id() -> Optional[int]:
        """Return a constant pseudo-user id."""
        UserAuth.initialize_session()
        return st.session_state.user_id

    @staticmethod
    def is_logged_in() -> bool:
        """
        Always 'True' so the app never blocks behind a login form.
        If your main app checks this to gate content, it will pass.
        """
        UserAuth.initialize_session()
        return True

    @staticmethod
    def login_user(username: str) -> bool:
        """
        Accept any username and set it in session. No DB or validation.
        This keeps compatibility if something still calls login_user().
        """
        UserAuth.initialize_session()
        u = (username or "").strip() or UserAuth.DEFAULT_USERNAME
        st.session_state.user_id = UserAuth.DEFAULT_USER_ID
        st.session_state.username = u
        st.session_state.user_data = {
            "id": UserAuth.DEFAULT_USER_ID,
            "username": u,
            "created_at": None,
        }
        # No DB load; just ensure local state exists
        UserAuth.load_user_data()
        return True

    @staticmethod
    def logout_user():
        """
        'Logout' just resets to the local pseudo-user and clears volatile state.
        No DB save, but we leave watchlist/portfolio unless you want a hard reset.
        """
        # If you want a full reset, uncomment the wipes below.
        # for key in ["portfolio", "alerts", "alert_history", "watchlist", "preferences"]:
        #     if key in st.session_state:
        #         del st.session_state[key]

        st.session_state.user_id = UserAuth.DEFAULT_USER_ID
        st.session_state.username = UserAuth.DEFAULT_USERNAME
        st.session_state.user_data = {
            "id": UserAuth.DEFAULT_USER_ID,
            "username": UserAuth.DEFAULT_USERNAME,
            "created_at": None,
        }
        st.success("Session reset")
        st.rerun()

    @staticmethod
    def load_user_data():
        """
        No-op for DB; ensures required session keys exist.
        """
        UserAuth.initialize_session()

    @staticmethod
    def save_current_state():
        """
        No-op: previously persisted to DB. Left for compatibility.
        """
        return

    # ---------- "Auto-save" shims (session only) ----------

    @staticmethod
    def auto_save_portfolio_position(symbol: str, shares: float, purchase_price: float, purchase_date):
        """
        Store a position locally in session. Returns True if appended.
        """
        UserAuth.initialize_session()
        try:
            st.session_state.portfolio.append(
                {
                    "id": None,  # no DB id
                    "symbol": symbol,
                    "shares": float(shares),
                    "purchase_price": float(purchase_price),
                    "purchase_date": purchase_date,
                    "purchase_value": float(shares) * float(purchase_price),
                }
            )
            return True
        except Exception:
            return False

    @staticmethod
    def auto_save_alert(alert_data: Dict):
        """
        Store alerts locally in session. Returns True if appended.
        """
        UserAuth.initialize_session()
        try:
            st.session_state.alerts.append(alert_data)
            # Mirror to alert_history for UI that expects it
            st.session_state.alert_history.append(alert_data)
            return True
        except Exception:
            return False

    @staticmethod
    def auto_save_watchlist_symbol(symbol: str, action: str = "add"):
        """
        Add/remove from session watchlist.
        """
        UserAuth.initialize_session()
        s = (symbol or "").strip().upper()
        if not s:
            return False

        wl = st.session_state.watchlist
        if action == "add":
            if s not in wl:
                wl.append(s)
                return True
            return False
        elif action == "remove":
            if s in wl:
                wl.remove(s)
                return True
            return False
        return False

    @staticmethod
    def auto_save_preference(preference_type: str, preference_key: str, preference_value):
        """
        Store a preference in session under a composite key.
        """
        UserAuth.initialize_session()
        st.session_state.preferences[(preference_type, preference_key)] = preference_value
        return True

    @staticmethod
    def get_user_preference(preference_type: str, preference_key: str, default_value=None):
        """
        Read a preference from session; return default if missing.
        """
        UserAuth.initialize_session()
        return st.session_state.preferences.get((preference_type, preference_key), default_value)

    # ---------- UI shims (kept so existing calls won't crash) ----------

    @staticmethod
    def display_login_form():
        """
        No real login anymore. Show a tiny info panel so existing
        calls don't break and users know it's session-only.
        """
        UserAuth.initialize_session()
        with st.expander("Session Profile (no login required)"):
            st.markdown(
                f"**User:** `{st.session_state.username}`  \n"
                "This build runs entirely in your local Streamlit session (no database)."
            )
            # Optional: allow renaming the session "username"
            new_name = st.text_input("Session name (optional)", value=st.session_state.username)
            if st.button("Update session name"):
                UserAuth.login_user(new_name)
                st.success("Session name updated")
                st.rerun()

    @staticmethod
    def display_user_menu():
        """
        Sidebar info so pages that expect a user menu won't fail.
        """
        UserAuth.initialize_session()
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"👤 **Session:** `{st.session_state.username}`")
        st.session_state.auto_save_enabled = st.sidebar.checkbox(
            "💾 Auto-save (session only)",
            value=st.session_state.get("auto_save_enabled", True),
            help="Saves to this Streamlit session state (no database).",
        )
        if st.sidebar.button("Reset session"):
            UserAuth.logout_user()

# Instantiate for `from modules.user_auth import user_auth`
user_auth = UserAuth()
