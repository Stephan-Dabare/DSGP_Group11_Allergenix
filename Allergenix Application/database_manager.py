import sqlite3
import bcrypt

class UserDatabase:
    def __init__(self, db_name="users.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        """Creates the users table if it does not exist."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )""")
        self.conn.commit()

    def add_user(self, username, email, password):
        """Hashes password and adds a new user."""
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        try:
            self.cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                                (username, email, hashed_password))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Username or email already exists

    def verify_user(self, email, password):
        """Checks if the email and password match."""
        self.cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
        result = self.cursor.fetchone()
        if result:
            stored_hashed_password = result[0]
            return bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8'))
        return False

# Initialize database
db = UserDatabase()
