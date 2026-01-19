import psycopg2
from config import Config


def get_db_connection():
    return psycopg2.connect(Config.DATABASE_URL)


# --- ADD THIS NEW FUNCTION ---
def get_scan_by_id(scan_id, user_auth0_id):
    """Fetches a single scan by its ID, ensuring it belongs to the logged-in user."""
    conn = get_db_connection()
    cur = conn.cursor()
    # Join with users table for security, ensuring the scan belongs to the current user
    cur.execute("""
        SELECT s.id, s.filename, s.file_path, s.gpu_predictions, s.overall_fake_probability
        FROM scans s
        JOIN users u ON s.user_id = u.id
        WHERE s.id = %s AND u.auth0_user_id = %s
    """, (scan_id, user_auth0_id))
    scan = cur.fetchone()
    cur.close()
    conn.close()
    if scan:
        # Convert the database row into a dictionary for easier use in the HTML template
        return {
            'id': scan[0],
            'filename': scan[1],
            'file_path': scan[2],
            'gpu_predictions': scan[3],
            'overall_fake_probability': scan[4]
        }
    return None


# --- END OF NEW FUNCTION ---

def create_tables():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS approved_doctors (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        hospital_name VARCHAR(255),
        status VARCHAR(20) DEFAULT 'active'
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        doctor_id VARCHAR(10) UNIQUE NOT NULL,
        auth0_user_id VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS scans (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        filename VARCHAR(255) NOT NULL,
        file_path VARCHAR(255) NOT NULL,
        gpu_predictions JSON,
        colab_predictions JSON,
        overall_fake_probability REAL,
        processed BOOLEAN DEFAULT FALSE
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS reports (
        id SERIAL PRIMARY KEY,
        scan_id INTEGER REFERENCES scans(id),
        patient_name VARCHAR(255),
        patient_id VARCHAR(100),
        doctor_name VARCHAR(255),
        hospital_name VARCHAR(255),
        email_sent_to VARCHAR(255)
    )''')

    conn.commit()
    cur.close()
    conn.close()
    print("Database tables created!")


def generate_doctor_id():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT doctor_id FROM users ORDER BY doctor_id DESC LIMIT 1")
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        last_id = int(result[0][2:])
        new_id = f"DR{last_id + 1:06d}"
    else:
        new_id = "DR000001"
    return new_id


def is_doctor_approved(email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM approved_doctors WHERE email = %s AND status = 'active'", (email,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result is not None


def add_approved_doctor(email, hospital_name=""):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO approved_doctors (email, hospital_name) VALUES (%s, %s)",
                    (email, hospital_name))
        conn.commit()
        return True
    except:
        return False
    finally:
        cur.close()
        conn.close()
