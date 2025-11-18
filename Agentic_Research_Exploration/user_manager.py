import sqlite3
import re
from typing import Dict, Optional, Tuple
from config import DB_PATH


class UserManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the users table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')

        conn.commit()
        conn.close()

    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def validate_name(self, name: str) -> bool:
        """Validate name (non-empty and reasonable length)."""
        return len(name.strip()) >= 2 and len(name.strip()) <= 100

    def register_user(self, session_id: str, name: str, email: str) -> Tuple[bool, str]:
        """Register a new user. Returns (success, message)."""
        name = name.strip()
        email = email.strip().lower()

        # Validate inputs
        if not self.validate_name(name):
            return False, "Name must be between 2 and 100 characters."

        if not self.validate_email(email):
            return False, "Please enter a valid email address."

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if email already exists
            cursor.execute("SELECT session_id FROM users WHERE email = ?", (email,))
            existing = cursor.fetchone()

            if existing:
                if existing[0] == session_id:
                    # Same session, update the user
                    cursor.execute("""
                        UPDATE users SET name = ?, registered_at = CURRENT_TIMESTAMP
                        WHERE session_id = ?
                    """, (name, session_id))
                    conn.commit()
                    conn.close()
                    return True, "User information updated successfully."
                else:
                    conn.close()
                    return False, "This email is already registered with another session."

            # Register new user
            cursor.execute("""
                INSERT INTO users (session_id, name, email)
                VALUES (?, ?, ?)
            """, (session_id, name, email))

            conn.commit()
            conn.close()
            return True, "User registered successfully."

        except sqlite3.IntegrityError:
            return False, "This email is already registered."
        except Exception as e:
            return False, f"Registration failed: {str(e)}"

    def get_user(self, session_id: str) -> Optional[Dict[str, str]]:
        """Get user information by session ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name, email, registered_at, is_active
                FROM users WHERE session_id = ? AND is_active = 1
            """, (session_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "name": row[0],
                    "email": row[1],
                    "registered_at": row[2],
                    "is_active": bool(row[3])
                }
            return None

        except Exception as e:
            print(f"Error getting user: {e}")
            return None

    def is_registered(self, session_id: str) -> bool:
        """Check if a session ID is registered."""
        return self.get_user(session_id) is not None

    def send_registration_notification(self, name: str, email: str, push_tool_func) -> None:
        """Send push notification about new user registration."""
        try:
            message = f"New user registered: {name} ({email})"
            push_tool_func(message)
        except Exception as e:
            print(f"Failed to send registration notification: {e}")
