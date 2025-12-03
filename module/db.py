import sqlite3

class Database:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
    
    def init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                room_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def get_history(self, session_id, room_id):
        self.cursor.execute(
            "SELECT role, message FROM chat_history WHERE session_id = ? AND room_id = ? ORDER BY timestamp ASC", 
            (session_id, room_id)
        )
        history = self.cursor.fetchall()
        # JANGAN tutup koneksi di sini!
        return history

    def save_message(self, session_id, room_id, role, message):
        self.cursor.execute('''
            INSERT INTO chat_history (session_id, room_id, role, message)
            VALUES (?, ?, ?, ?)
        ''', (session_id, room_id, role, message))
        self.conn.commit()
        # JANGAN tutup koneksi di sini!
    
    def close(self):
        """Method untuk menutup koneksi ketika sudah selesai"""
        self.conn.close()