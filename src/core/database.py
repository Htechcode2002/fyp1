import mysql.connector
from mysql.connector import Error
from src.core.config_manager import ConfigManager
import threading
import queue
import time
from datetime import datetime

class DatabaseManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.config = ConfigManager().get("db")
            cls._instance.last_error_time = 0
            cls._instance.error_throttle_seconds = 60 # Only print same error every 60s
            
            # Async Worker Setup
            # CRITICAL: Set max queue size to prevent memory overflow
            cls._instance.queue = queue.Queue(maxsize=1000)  # Limit to 1000 pending events
            cls._instance.running = True
            cls._instance.worker_thread = threading.Thread(target=cls._instance._process_queue, daemon=True)
            cls._instance.worker_thread.start()
            
        return cls._instance

    def connect(self):
        """
        Creates and returns a FRESH connection to the database.
        Crucial: Never share connections across threads.
        """
        try:
            db_config = self.config
            # Create a localized connection object
            conn = mysql.connector.connect(
                host=db_config.get("host"),
                port=db_config.get("port", 4000),
                user=db_config.get("user"),
                password=db_config.get("password"),
                database=db_config.get("database", "test"),
                connection_timeout=20, # Shorter timeout to fail fast
                autocommit=True,       # Simplify transactions
                ssl_disabled=False     # Security required
            )
            return conn
        except Error as e:
            print(f"[DB] ❌ Threaded connection failed: {e}")
            return None

    def _process_queue(self):
        """Background worker to process events from the queue with batch insert."""
        batch = []
        batch_size = 10  # Insert up to 10 events at once
        last_insert_time = time.time()
        
        while self.running:
            try:
                # Block for 0.5s to allow batching
                event_data = self.queue.get(timeout=0.5)
                batch.append(event_data)
                self.queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[DB] Worker Error: {e}")
                continue
            
            # Insert batch if full or timeout (2 seconds)
            current_time = time.time()
            if len(batch) >= batch_size or (len(batch) > 0 and current_time - last_insert_time > 2.0):
                self._insert_batch(batch)
                batch = []
                last_insert_time = current_time
        
        # Flush remaining on shutdown
        if batch:
            self._insert_batch(batch)
    
    def _insert_batch(self, batch):
        """Insert a batch with retry, or save to local file if failed."""
        if not batch:
            return
        
        # Check cooling period - if active, save directly to local to avoid network lag
        if hasattr(self, '_cooling_until') and time.time() < self._cooling_until:
            self._save_to_local(batch)
            return
            
        max_retries = 1 # Only try once in real-time to keep things fast
        try:
            conn = self.connect()
            if conn:
                cursor = conn.cursor()
                query = """
                INSERT INTO crossing_events (video_id, location, line_name, count_left, count_right, clothing_color, gender, age, mask_status, handbag, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(query, batch)
                conn.commit()
                cursor.close()
                conn.close()
                
                # Success! Now try to sync old local data if any
                self._sync_local_cache()
                return 
            else:
                raise Exception("Network connection unavailable")
        except Exception as e:
            print(f"[DB] ⚠️ Network Error: {e}. Saving data to local buffer.")
            self._save_to_local(batch)
            # Enter short cooling if network error
            self._cooling_until = time.time() + 60 # 1 minute cooling

    def _save_to_local(self, batch):
        """Save failed batch to a local JSON file."""
        import json
        import os
        try:
            buffer_path = "offline_events_buffer.json"
            existing_data = []
            if os.path.exists(buffer_path):
                with open(buffer_path, 'r') as f:
                    try: existing_data = json.load(f)
                    except: existing_data = []
            
            existing_data.extend(batch)
            # Limit buffer size to 10,000 events to prevent massive files
            if len(existing_data) > 10000:
                existing_data = existing_data[-10000:]
                
            with open(buffer_path, 'w') as f:
                json.dump(existing_data, f)
        except Exception as e:
            print(f"[DB] ❌ CRITICAL: Failed to save local buffer: {e}")

    def _sync_local_cache(self):
        """Try to upload locally buffered data to the cloud."""
        import json
        import os
        buffer_path = "offline_events_buffer.json"
        if not os.path.exists(buffer_path):
            return

        print(f"[DB] 🔄 Found offline data. Attempting to sync to cloud...")
        try:
            with open(buffer_path, 'r') as f:
                offline_data = json.load(f)
            
            if not offline_data:
                os.remove(buffer_path)
                return

            conn = self.connect()
            if conn:
                cursor = conn.cursor()
                query = """
                INSERT INTO crossing_events (video_id, location, line_name, count_left, count_right, clothing_color, gender, age, mask_status, handbag, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                # Process in small chunks to avoid huge transactions
                for i in range(0, len(offline_data), 50):
                    chunk = offline_data[i:i+50]
                    cursor.executemany(query, chunk)
                
                conn.commit()
                cursor.close()
                conn.close()
                
                # Clear buffer on success
                os.remove(buffer_path)
                print(f"[DB] ✅ Successfully synced {len(offline_data)} offline events to cloud.")
        except Exception as e:
            print(f"[DB] ⏳ Sync failed, will try again later: {e}")

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
                gender VARCHAR(50),
                age VARCHAR(50),
                mask_status VARCHAR(50),
                handbag TINYINT DEFAULT 0,
                timestamp DATETIME
            )
            """
            cursor.execute(query)
            conn.commit()

            # Check for gender and age columns and add if missing
            cursor.execute("SHOW COLUMNS FROM crossing_events LIKE 'gender'")
            if cursor.fetchone() is None:
                print("Adding missing 'gender' column...")
                cursor.execute("ALTER TABLE crossing_events ADD COLUMN gender VARCHAR(50) AFTER clothing_color")
                conn.commit()

            cursor.execute("SHOW COLUMNS FROM crossing_events LIKE 'age'")
            col_info = cursor.fetchone()
            if col_info is None:
                print("Adding missing 'age' column...")
                cursor.execute("ALTER TABLE crossing_events ADD COLUMN age VARCHAR(50) AFTER gender")
                conn.commit()
            elif "int" in col_info[1].lower():
                print("Converting 'age' column from INT to VARCHAR(50)...")
                cursor.execute("ALTER TABLE crossing_events MODIFY COLUMN age VARCHAR(50)")
                conn.commit()

            # Check for mask_status column and add if missing
            cursor.execute("SHOW COLUMNS FROM crossing_events LIKE 'mask_status'")
            if cursor.fetchone() is None:
                print("Adding missing 'mask_status' column...")
                cursor.execute("ALTER TABLE crossing_events ADD COLUMN mask_status VARCHAR(50) AFTER age")
                conn.commit()

            # Check for handbag column and add if missing
            cursor.execute("SHOW COLUMNS FROM crossing_events LIKE 'handbag'")
            if cursor.fetchone() is None:
                print("Adding missing 'handbag' column...")
                cursor.execute("ALTER TABLE crossing_events ADD COLUMN handbag TINYINT DEFAULT 0 AFTER mask_status")
                conn.commit()

            print("Table 'crossing_events' check/creation successful.")
        except Error as e:
            print(f"Error creating table: {e}")
        finally:
            if conn and conn.is_connected():
                cursor.close()

    def insert_event(self, video_id, location, line_name, count_left, count_right, clothing_color, gender=None, age=None, mask_status=None, handbag=0):
        """Queue a crossing event for insertion."""
        # Use local PC time instead of database server time
        local_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # CRITICAL: Use non-blocking put with timeout to prevent crashes
        try:
            self.queue.put((video_id, location, line_name, count_left, count_right, clothing_color, gender, age, mask_status, handbag, local_timestamp), block=False)
        except queue.Full:
            print(f"⚠️ WARNING: Database queue is full ({self.queue.qsize()} events). Dropping event to prevent memory overflow.")

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

    def delete_event(self, event_id):
        """Delete a single event by ID"""
        conn = self.connect()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            query = "DELETE FROM crossing_events WHERE id = %s"
            cursor.execute(query, (event_id,))
            conn.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Delete Error: {e}")
            return False

    def delete_events_by_filter(self, filters):
        """Delete multiple events based on filters"""
        conn = self.connect()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            query = "DELETE FROM crossing_events WHERE 1=1"
            params = []

            # DateTime range filter
            if filters.get('start_datetime'):
                query += " AND timestamp >= %s"
                params.append(filters['start_datetime'])
            if filters.get('end_datetime'):
                query += " AND timestamp <= %s"
                params.append(filters['end_datetime'])

            # Video ID filter
            if filters.get('video_id'):
                query += " AND video_id = %s"
                params.append(filters['video_id'])

            # Gender filter
            if filters.get('gender') and filters['gender'] != 'All':
                query += " AND gender = %s"
                params.append(filters['gender'])

            # Color filter
            if filters.get('color') and filters['color'] != 'All':
                query += " AND clothing_color = %s"
                params.append(filters['color'])

            # Mask filter
            if filters.get('mask') and filters['mask'] != 'All':
                query += " AND mask_status = %s"
                params.append(filters['mask'])

            # Handbag filter
            if filters.get('handbag') and filters['handbag'] != 'All':
                handbag_value = 1 if filters['handbag'] == 'With Handbag' else 0
                query += " AND handbag = %s"
                params.append(handbag_value)

            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            return deleted_count
        except Error as e:
            print(f"Bulk Delete Error: {e}")
            return False

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
