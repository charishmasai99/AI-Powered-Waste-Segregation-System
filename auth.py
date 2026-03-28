# =============================================================================
# auth.py  ─  EcoSort AI  |  Complete Authentication System
# Handles: Register, Login (email+password), Google OAuth, Session management
# =============================================================================

import json, os, hashlib, secrets, re
from datetime import datetime
import streamlit as st

# ── Users database (JSON file) ────────────────────────────────────────────────
USERS_FILE = "users.json"

def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return hashed, salt

def _validate_email(email: str) -> bool:
    return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-z]{2,}$", email.lower()))

def _validate_password(password: str) -> list:
    """Returns list of unmet requirements."""
    issues = []
    if len(password) < 8:
        issues.append("At least 8 characters")
    if not re.search(r"[A-Z]", password):
        issues.append("At least one uppercase letter")
    if not re.search(r"[0-9]", password):
        issues.append("At least one number")
    return issues

# ── Register ──────────────────────────────────────────────────────────────────
def register_user(name: str, email: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)"""
    if not name.strip():
        return False, "Name cannot be empty."
    if not _validate_email(email):
        return False, "Please enter a valid email address."
    issues = _validate_password(password)
    if issues:
        return False, "Password must have: " + ", ".join(issues) + "."

    users = _load_users()
    if email.lower() in users:
        return False, "An account with this email already exists. Please sign in."

    hashed, salt = _hash_password(password)
    users[email.lower()] = {
        "name": name.strip(),
        "email": email.lower(),
        "password_hash": hashed,
        "salt": salt,
        "picture": f"https://ui-avatars.com/api/?name={name.replace(' ','+')}&background=1E8E3E&color=fff&size=64",
        "provider": "email",
        "created_at": datetime.now().isoformat(),
    }
    _save_users(users)
    return True, "Account created successfully!"

# ── Login ─────────────────────────────────────────────────────────────────────
def login_user(email: str, password: str) -> tuple[bool, str, dict]:
    """Returns (success, message, user_info)"""
    if not email or not password:
        return False, "Please enter your email and password.", {}

    users = _load_users()
    user = users.get(email.lower())

    if not user:
        return False, "No account found with this email. Please register first.", {}
    if user.get("provider") == "google":
        return False, "This email is linked to Google. Please use 'Sign in with Google'.", {}

    hashed, _ = _hash_password(password, user["salt"])
    if hashed != user["password_hash"]:
        return False, "Incorrect password. Please try again.", {}

    return True, "Welcome back!", {
        "name": user["name"],
        "email": user["email"],
        "picture": user["picture"],
        "provider": "email",
    }

# ── Google OAuth user upsert ──────────────────────────────────────────────────
def upsert_google_user(google_info: dict) -> dict:
    """Creates or updates a Google-authenticated user, returns user_info."""
    users = _load_users()
    email = google_info.get("email", "").lower()
    name  = google_info.get("name", "EcoUser")
    pic   = google_info.get("picture", f"https://ui-avatars.com/api/?name={name.replace(' ','+')}&background=1E8E3E&color=fff&size=64")

    if email not in users:
        users[email] = {
            "name": name,
            "email": email,
            "password_hash": None,
            "salt": None,
            "picture": pic,
            "provider": "google",
            "created_at": datetime.now().isoformat(),
        }
        _save_users(users)
    else:
        # Update name/picture from Google in case they changed
        users[email]["name"]    = name
        users[email]["picture"] = pic
        _save_users(users)

    return {"name": name, "email": email, "picture": pic, "provider": "google"}


# =============================================================================
# AUTH UI  ─  renders the full Login / Register / Google screen
# =============================================================================
def render_auth_ui():
    """
    Call this function where you previously had the Google sign-in block.
    It handles the full auth flow and sets st.session_state.user_info on success.
    """

    # ── Google OAuth callback check ───────────────────────────────────────────
    try:
        from streamlit_google_auth import Authenticate
        GOOGLE_AUTH_AVAILABLE = True
    except ImportError:
        GOOGLE_AUTH_AVAILABLE = False

    if GOOGLE_AUTH_AVAILABLE:
        try:
            redirect_uri = st.secrets.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")
            cookie_key   = st.secrets.get("COOKIE_KEY", "ecosort-secret-key-32chars-xyz!!")

            authenticator = Authenticate(
                secret_credentials_path="google_credentials.json",
                cookie_name="ecosort_auth",
                cookie_key=cookie_key,
                redirect_uri=redirect_uri,
            )
            authenticator.check_authentification()

            if st.session_state.get("connected"):
                g_info = st.session_state.get("user_info", {})
                user   = upsert_google_user(g_info)
                st.session_state.user_info = user
                st.rerun()
        except Exception:
            GOOGLE_AUTH_AVAILABLE = False

    # ── Session tab state ─────────────────────────────────────────────────────
    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "login"   # "login" | "register"

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .auth-wrap {
        min-height: 100vh;
        display: flex; align-items: center; justify-content: center;
        padding: 40px 24px;
    }
    .auth-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(74,222,128,0.2);
        border-radius: 28px; padding: 44px 44px 36px;
        max-width: 460px; width: 100%;
        backdrop-filter: blur(16px);
        box-shadow: 0 24px 64px rgba(0,0,0,0.4);
        margin: 0 auto;
    }
    .auth-logo  { font-size: 46px; text-align: center; margin-bottom: 10px; }
    .auth-title { font-size: 26px; font-weight: 900; color: #FFFFFF !important;
                  text-align: center; margin-bottom: 6px; }
    .auth-sub   { font-size: 13px; color: rgba(255,255,255,0.45) !important;
                  text-align: center; margin-bottom: 28px; line-height: 1.6; }
    .auth-divider {
        display: flex; align-items: center; gap: 12px; margin: 18px 0;
    }
    .auth-divider hr { flex:1; border:none; border-top:1px solid rgba(255,255,255,0.12); }
    .auth-divider span { font-size:12px; color:rgba(255,255,255,0.3); font-weight:500; }
    .auth-footer { font-size:11px; color:rgba(255,255,255,0.22);
                   text-align:center; margin-top:22px; line-height:1.7; }
    .tab-row { display:flex; gap:0; margin-bottom:28px;
               background:rgba(255,255,255,0.05); border-radius:12px; padding:4px; }
    .tab-btn { flex:1; padding:9px; border-radius:9px; border:none;
               font-size:14px; font-weight:600; cursor:pointer; transition:all 0.2s; }
    .tab-active   { background:#1E8E3E; color:#fff; }
    .tab-inactive { background:transparent; color:rgba(255,255,255,0.4); }

    /* Input fields */
    [data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(74,222,128,0.2) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-size: 14px !important;
        padding: 10px 14px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: rgba(74,222,128,0.6) !important;
        box-shadow: 0 0 0 2px rgba(74,222,128,0.15) !important;
    }
    [data-testid="stTextInput"] label {
        color: rgba(255,255,255,0.7) !important;
        font-size: 13px !important; font-weight: 500 !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg,#1E8E3E,#16A34A) !important;
        color: #fff !important; border: none !important;
        border-radius: 12px !important; font-size: 15px !important;
        font-weight: 700 !important; height: 50px !important;
        box-shadow: 0 4px 16px rgba(30,142,62,0.35) !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 28px rgba(30,142,62,0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Google button */
    .google-signin-btn {
        display: flex; align-items: center; justify-content: center; gap: 10px;
        background: #FFFFFF; color: #1F1F1F; border: none; border-radius: 12px;
        padding: 13px 20px; font-size: 15px; font-weight: 600;
        cursor: pointer; width: 100%; transition: all 0.2s;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        font-family: Inter, sans-serif; text-decoration: none;
        margin-bottom: 4px;
    }
    .google-signin-btn:hover {
        box-shadow: 0 8px 28px rgba(0,0,0,0.4);
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Layout: centered card ─────────────────────────────────────────────────
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="auth-logo">♻️</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">EcoSort AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Sort smarter. Live greener.<br>Your eco journey starts here.</div>', unsafe_allow_html=True)

        # ── Tab switcher ──────────────────────────────────────────────────────
        tab_col1, tab_col2 = st.columns(2)
        with tab_col1:
            if st.button("🔑  Sign In", use_container_width=True,
                         type="primary" if st.session_state.auth_tab == "login" else "secondary"):
                st.session_state.auth_tab = "login"
                st.rerun()
        with tab_col2:
            if st.button("✨  Create Account", use_container_width=True,
                         type="primary" if st.session_state.auth_tab == "register" else "secondary"):
                st.session_state.auth_tab = "register"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # LOGIN TAB
        # ══════════════════════════════════════════════════════════════════════
        if st.session_state.auth_tab == "login":
            with st.form("login_form", clear_on_submit=False):
                email    = st.text_input("📧  Email address", placeholder="you@example.com")
                password = st.text_input("🔒  Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                success, msg, user = login_user(email, password)
                if success:
                    st.session_state.user_info = user
                    st.success(f"✅ Welcome back, {user['name']}!")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

            # Forgot password (UI only — extend as needed)
            st.markdown("""
            <div style="text-align:right;margin-top:-8px;margin-bottom:12px">
              <span style="font-size:12px;color:rgba(74,222,128,0.7);cursor:pointer">
                Forgot password?
              </span>
            </div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # REGISTER TAB
        # ══════════════════════════════════════════════════════════════════════
        else:
            with st.form("register_form", clear_on_submit=False):
                name      = st.text_input("👤  Full name", placeholder="Your name")
                email     = st.text_input("📧  Email address", placeholder="you@example.com")
                password  = st.text_input("🔒  Password", type="password",
                                          placeholder="Min 8 chars, 1 uppercase, 1 number")
                password2 = st.text_input("🔒  Confirm password", type="password",
                                          placeholder="Repeat your password")
                submitted  = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if submitted:
                if password != password2:
                    st.error("❌ Passwords do not match.")
                else:
                    success, msg = register_user(name, email, password)
                    if success:
                        st.success(f"✅ {msg} Please sign in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            st.markdown("""
            <div style="background:rgba(74,222,128,0.07);border-radius:10px;
                        padding:10px 14px;margin-top:4px">
              <div style="font-size:12px;color:rgba(255,255,255,0.5);line-height:1.8">
                🔐 Password must have:<br>
                &nbsp;&nbsp;• At least 8 characters<br>
                &nbsp;&nbsp;• At least one uppercase letter (A–Z)<br>
                &nbsp;&nbsp;• At least one number (0–9)
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Divider ───────────────────────────────────────────────────────────
        st.markdown("""
        <div class="auth-divider">
          <hr/><span>OR</span><hr/>
        </div>""", unsafe_allow_html=True)

        # ── Google Sign-In button ─────────────────────────────────────────────
        if GOOGLE_AUTH_AVAILABLE:
            if st.button("🔵  Continue with Google", use_container_width=True):
                try:
                    authenticator.login()
                except Exception as e:
                    st.error(f"Google sign-in error: {e}")
        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
                        border-radius:12px;padding:13px;text-align:center">
              <span style="font-size:13px;color:rgba(255,255,255,0.3)">
                🔵 Google Sign-In unavailable (check google_credentials.json)
              </span>
            </div>""", unsafe_allow_html=True)

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="auth-footer">
          By continuing, you agree to EcoSort's Terms of Service.<br>
          Your data is never shared or sold. 🌱
        </div>""", unsafe_allow_html=True)