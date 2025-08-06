import sqlite3
import logging
from typing import List, Dict, Optional


class DatabaseUtil:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = None
        self.logger = logging.getLogger(__name__)
        self._setup_database()

    def _setup_database(self):
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like row access
            self.connection.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            self._create_tables()
            self.logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error setting up database: {e}")
            raise

    def _create_tables(self):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stocks (
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                PRIMARY KEY (user_id, symbol),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stocks_user_id ON stocks(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

        self.connection.commit()
        self.logger.info("Database tables created successfully")

    def close(self):
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")

    def create_user(self, username: str, email: str) -> int:
        cursor = self.connection.cursor()
        try:
            cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", (username, email))
            self.connection.commit()
            user_id = cursor.lastrowid
            self.logger.info(f"Created user: {username} with ID: {user_id}")
            return user_id
        except sqlite3.IntegrityError as e:
            self.logger.error(f"Error creating user {username}: {e}")
            raise

    def get_user(self, user_id: int) -> Optional[Dict]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_user(self, user_id: int, username: str = None, email: str = None) -> bool:
        if not username and not email:
            return False

        cursor = self.connection.cursor()
        update_fields = []
        values = []

        if username:
            update_fields.append("username = ?")
            values.append(username)
        if email:
            update_fields.append("email = ?")
            values.append(email)

        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(user_id)

        query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, values)
        self.connection.commit()

        updated = cursor.rowcount > 0
        if updated:
            self.logger.info(f"Updated user ID: {user_id}")
        return updated

    def delete_user(self, user_id: int) -> bool:
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.connection.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.info(f"Deleted user ID: {user_id}")
        return deleted

    def add_stock(self, user_id: int, symbol: str) -> bool:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO stocks (user_id, symbol) VALUES (?, ?)",
                (user_id, symbol.upper()),
            )
            self.connection.commit()
            added = cursor.rowcount > 0
            if added:
                self.logger.info(f"Added stock {symbol} for user ID: {user_id}")
            return added
        except sqlite3.IntegrityError as e:
            self.logger.error(f"Error adding stock {symbol} for user {user_id}: {e}")
            raise

    def has_stock(self, user_id: int, symbol: str) -> bool:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT 1 FROM stocks WHERE user_id = ? AND symbol = ?", (user_id, symbol.upper())
        )
        return cursor.fetchone() is not None

    def get_user_stocks(self, user_id: int) -> List[Dict]:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM stocks 
            WHERE user_id = ? 
            ORDER BY symbol
        """,
            (user_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_stocks_by_symbol(self, symbol: str) -> List[Dict]:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT s.*, u.username, u.email
            FROM stocks s
            JOIN users u ON s.user_id = u.id
            WHERE s.symbol = ?
            ORDER BY u.username
        """,
            (symbol.upper(),),
        )
        return [dict(row) for row in cursor.fetchall()]

    def remove_stock(self, user_id: int, symbol: str) -> bool:
        cursor = self.connection.cursor()
        cursor.execute(
            "DELETE FROM stocks WHERE user_id = ? AND symbol = ?", (user_id, symbol.upper())
        )
        self.connection.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.info(f"Removed stock {symbol} for user ID: {user_id}")
        return deleted

    def get_all_users(self) -> List[Dict]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at")
        return [dict(row) for row in cursor.fetchall()]

    def get_all_stocks(self) -> List[Dict]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM stocks ORDER BY user_id, symbol")
        return [dict(row) for row in cursor.fetchall()]

    def get_user_stock_count(self, user_id: int) -> int:
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks WHERE user_id = ?", (user_id,))
        return cursor.fetchone()[0]

    def get_stock_users_count(self, symbol: str) -> int:
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks WHERE symbol = ?", (symbol.upper(),))
        return cursor.fetchone()[0]

    def clear_all_data(self):
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM stocks")
        cursor.execute("DELETE FROM users")
        self.connection.commit()
        self.logger.info("Cleared all data from database")
