import sqlite3

# Path to your SQLite database file
db_path = 'app.db'

# Connect to the database
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
''')

# List all users and their roles
c.execute("SELECT username, role FROM users")
users = c.fetchall()

if users:
    print("List of users:")
    for user in users:
        print(f"Username: {user[0]}, Role: {user[1]}")
else:
    print("No users found in the database.")

# Close the connection
conn.close()
