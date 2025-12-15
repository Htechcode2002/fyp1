import mysql.connector
from mysql.connector import Error
from src.core.config_manager import ConfigManager
import threading
import queue
import time

class DatabaseManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.connection = None
            cls._instance.config = ConfigManager().get("db")
            
            # Async Worker Setup
            cls._instance.queue = queue.Queue()
            cls._instance.running = True
            cls._instance.worker_thread = threading.Thread(target=cls._instance._process_queue, daemon=True)
            cls._instance.worker_thread.start()
            
        return cls._instance

    def connect(self):
        """Establish connection to the database."""
        # Check if existing connection is valid
        if self.connection:
            try:
                if self.connection.is_connected():
                    return self.connection
            except:
                pass # Connection stale/broken

        try:
            db_config = self.config
            self.connection = mysql.connector.connect(
                host=db_config.get("host"),
                port=db_config.get("port", 4000),
                user=db_config.get("user"),
                password=db_config.get("password"),
                database=db_config.get("database", "test")
            )
            if self.connection.is_connected():
                # print("Successfully connected to the database")
                return self.connection
        except Error as e:
            print(f"Error connecting to database: {e}")
            return None

    def _process_queue(self):
        """Background worker to process events from the queue."""
        while self.running:
            try:
                # Block for 1s to allow check for self.running
                event_data = self.queue.get(timeout=1.0)
                
                # We have an event, ensure connection
                conn = self.connect()
                if conn:
                    try:
                        cursor = conn.cursor()
                        query = """
                        INSERT INTO crossing_events (video_id, location, line_name, count_left, count_right, clothing_color)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(query, event_data)
                        conn.commit()
                        cursor.close()
                    except Error as e:
                        print(f"DB Insert Error: {e}")
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"DB Worker Error: {e}")

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        # Create tables is usually run once at startup, okay to be sync
        conn = self.connect()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            query = """
            CREATE TABLE IF NOT EXISTS crossing_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255),
                location VARCHAR(255),
                line_name VARCHAR(255),
                count_left INT DEFAULT 0,
                count_right INT DEFAULT 0,
                clothing_color VARCHAR(255),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(query)
            conn.commit()

            # Check for video_id column and add if missing
            cursor.execute("SHOW COLUMNS FROM crossing_events LIKE 'video_id'")
            if cursor.fetchone() is None:
                print("Adding missing 'video_id' column...")
                cursor.execute("ALTER TABLE crossing_events ADD COLUMN video_id VARCHAR(255) AFTER id")
                conn.commit()
                
            print("Table 'crossing_events' check/creation successful.")
        except Error as e:
            print(f"Error creating table: {e}")
        finally:
            if conn and conn.is_connected():
                cursor.close()

    def insert_event(self, video_id, location, line_name, count_left, count_right, clothing_color):
        """Queue a crossing event for insertion."""
        # Non-blocking put
        self.queue.put((video_id, location, line_name, count_left, count_right, clothing_color))

    def get_analytics_data(self, hours=1, interval='minute'):
        """
        Fetch aggregated crossing counts for the last N hours.
        interval: 'minute', 'hour', 'day'
        """
        conn = self.connect()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Determine Date Format based on interval
            date_format = "%Y-%m-%d %H:%i:00" # Default minute
            if interval == 'hour':
                date_format = "%Y-%m-%d %H:00:00"
            elif interval == 'day':
                date_format = "%Y-%m-%d 00:00:00"

            query = f"""
            SELECT 
                DATE_FORMAT(timestamp, '{date_format}') as time_bucket,
                SUM(count_left + count_right) as total_count
            FROM crossing_events
            WHERE timestamp >= NOW() - INTERVAL %s HOUR
            GROUP BY time_bucket
            ORDER BY time_bucket ASC
            """
            cursor.execute(query, (hours,))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"Analytics Query Error: {e}")
            return []

    def execute_safe_query(self, sql):
        """
        Executes a raw SQL query for AI usage.
        STRICTLY READ-ONLY: Rejects inappropriate commands.
        """
        clean_sql = sql.strip().upper()
        if not clean_sql.startswith("SELECT"):
            print(f"Safety Block: Attempted non-SELECT query: {sql}")
            return None
            
        conn = self.connect()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"AI Query Execution Error: {e}")
            return str(e)

    def close(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            
        if self.connection and self.connection.is_connected():
            self.connection.close()
            # print("Database connection closed.")
            self.connection = None

if __name__ == "__main__":
    db = DatabaseManager()
    db.create_tables()
