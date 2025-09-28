
import sqlite3
import hashlib

from src.config import USERS_DB, USER_DATA_DIR
from src.LogHandler.log_config import get_logger

logger = get_logger(__name__)

DB_PATH = USERS_DB

def init_db():
    """Initializes the database and creates the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    """Creates a new user in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        create_user_artifacts(username)
        logger.info(f"User {username} created successfully.")
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"Attempt to create user {username} which already exists.")
        return False  # Username already exists
    finally:
        conn.close()

def login_user(username, password):
    """Logs in a user by verifying their credentials."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, password_hash))
    user = cursor.fetchone()
    conn.close()
    if user:
        logger.info(f"User {username} logged in successfully.")
        return "some_generated_token"  # In a real application, generate a proper token (e.g., JWT)
    else:
        logger.warning(f"Failed login attempt for user {username}.")
        return None

def create_user_artifacts(username):
    """Creates the model and metrics files for a new user."""
    user_dir = USER_DATA_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "model.pkl").touch()
    with open(user_dir / "metrics.json", "w") as f:
        f.write("{}")

if __name__ == '__main__':
    init_db()
    # Example usage:
    # create_user("testuser", "password123")
    # token = login_user("testuser", "password123")
    # print(f"Login successful, token: {token}")
