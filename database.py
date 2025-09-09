import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()

# make a new one if one doesn't alrdy exist
c.execute("""
CREATE TABLE IF NOT EXISTS Users (
    UserID INTEGER PRIMARY KEY AUTOINCREMENT,
    Username TEXT UNIQUE NOT NULL,
    Password TEXT NOT NULL
)
""")
conn.commit()

def add_user(username, password):
    try:
        c.execute("INSERT INTO Users (Username, Password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate(username, password):
    c.execute("SELECT * FROM Users WHERE Username=? AND Password=?", (username, password))
    return c.fetchone() is not None
