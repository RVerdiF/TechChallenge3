
import sqlite3
import hashlib
import os

DB_PATH = "users.db"

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
        return True
    except sqlite3.IntegrityError:
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
        return "some_generated_token"  # In a real application, generate a proper token (e.g., JWT)
    else:
        return None

def create_user_artifacts(username):
    """Creates the model and metrics files for a new user."""
    user_dir = os.path.join("user_data", username)
    os.makedirs(user_dir, exist_ok=True)
    with open(os.path.join(user_dir, "model.pkl"), "w") as f:
        f.write("")
    with open(os.path.join(user_dir, "metrics.json"), "w") as f:
        f.write("{}")

if __name__ == '__main__':
    init_db()
    # Example usage:
    # create_user("testuser", "password123")
    # token = login_user("testuser", "password123")
    # print(f"Login successful, token: {token}")
