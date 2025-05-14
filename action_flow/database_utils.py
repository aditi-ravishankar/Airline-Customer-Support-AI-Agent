import mysql.connector
import time
from .settings import DB_CONFIG

def connect_to_mysql(attempts=3, delay=2):
    """Establish a connection to MySQL with retry logic."""
    attempt = 1
    while attempt <= attempts:
        try:
            # Connect to the database with the specified config and plugin for authentication
            cnx = mysql.connector.connect(**DB_CONFIG, auth_plugin='mysql_native_password')
            print("✅ Connected to DB!")
            return cnx
        except (mysql.connector.Error, IOError) as err:
            if attempt == attempts:
                # If maximum attempts reached, return None
                print(f"Failed to connect, exiting without a connection: {err}")
                return None
            print(f"Connection failed: {err}. Retrying ({attempt}/{attempts})...")
            time.sleep(delay ** attempt)  # Exponential backoff
            attempt += 1
    return None

def is_db_connected(cnx):
    """Check if the DB connection is alive and return True/False."""
    if cnx and cnx.is_connected():
        with cnx.cursor() as cursor:
            cursor.execute("SELECT 1")  # Simple query to check connection
            return True
    else:
        return False

def select_record(query):
    """Fetch records from the database using a given query."""
    cnx = connect_to_mysql()
    if cnx is None:
        return []

    try:
        with cnx.cursor(buffered=True) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                print("No records found.")
            return rows
    except Exception as e:
        print(f"Failed to execute query: {e}")
        return []
    finally:
        if cnx:
            cnx.close()  # Ensure to close connection after use

def delete_record(query):
    """Delete records from the database using a given query."""
    cnx = connect_to_mysql()
    if cnx is None:
        print("No database connection.")
        return False

    try:
        with cnx.cursor() as cursor:
            cursor.execute(query)
            cnx.commit()
            print("Record deleted successfully.")
            return True
    except Exception as e:
        print(f"Failed to delete record: {e}")
        return False
    finally:
        if cnx:
            cnx.close()  # Ensure to close connection after use
