# this is database.py file
import sqlite3
import numpy as np
from datetime import datetime

def initialize_database():
    """Initialize the database with required tables."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()

    # Create Person table with a new column for captured face
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Person (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_embedding BLOB NOT NULL,
            embedding_count INTEGER DEFAULT 1,
            registration_date TEXT NOT NULL,
            captured_face TEXT  -- New column for storing the face image path
        )
    """)

    # Create EntryLog table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS EntryLog (
            serial_no INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            captured_face TEXT NOT NULL,
            location TEXT NOT NULL,
            in_time TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES Person(person_id)
        )
    """)

    conn.commit()
    conn.close()

def add_person(face_embedding, captured_face_path):
    """Add a new person to the database with the captured face."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()
    registration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO Person (face_embedding, registration_date, captured_face)
        VALUES (?, ?, ?)
    """, (face_embedding.tobytes(), registration_date, captured_face_path))
    person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return person_id

def get_all_persons(filter_date=None, start_date=None, end_date=None):
    """Retrieve all person records from the database."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()

    if filter_date:
        cursor.execute("""
            SELECT person_id, registration_date, embedding_count, captured_face
            FROM Person
            WHERE DATE(registration_date) = ?
        """, (filter_date,))
    elif start_date and end_date:
        cursor.execute("""
            SELECT person_id, registration_date, embedding_count, captured_face
            FROM Person
            WHERE DATE(registration_date) BETWEEN ? AND ?
        """, (start_date, end_date))
    else:
        cursor.execute("""
            SELECT person_id, registration_date, embedding_count, captured_face
            FROM Person
        """)

    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "registration_date": row[1],
            "embedding_count": row[2],
            "captured_face": row[3]
        }
        for row in rows
    ]


def refine_embedding(person_id, new_embedding):
    """Refine the existing embedding for a person."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()

    cursor.execute("SELECT face_embedding, embedding_count FROM Person WHERE person_id = ?", (person_id,))
    row = cursor.fetchone()
    if row:
        existing_embedding = np.frombuffer(row[0], dtype=np.float32)
        embedding_count = row[1]

        # Update embedding with weighted average
        refined_embedding = (existing_embedding * embedding_count + new_embedding) / (embedding_count + 1)
        refined_embedding_count = embedding_count + 1

        cursor.execute("""
            UPDATE Person 
            SET face_embedding = ?, embedding_count = ?
            WHERE person_id = ?
        """, (refined_embedding.tobytes(), refined_embedding_count, person_id))
        conn.commit()

    conn.close()

def get_logs(filter_date=None, start_date=None, end_date=None):
    """Retrieve logs from the database, optionally filtered by date."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()

    if filter_date:
        cursor.execute("""
            SELECT serial_no, person_id, in_time, captured_face
            FROM EntryLog
            WHERE DATE(in_time) = ?
        """, (filter_date,))
    elif start_date and end_date:
        cursor.execute("""
            SELECT serial_no, person_id, in_time, captured_face
            FROM EntryLog
            WHERE DATE(in_time) BETWEEN ? AND ?
        """, (start_date, end_date))
    else:
        cursor.execute("""
            SELECT serial_no, person_id, in_time, captured_face
            FROM EntryLog
        """)

    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "person_id": row[1],
            "timestamp": row[2],
            "captured_face": f"/captured_faces/{row[3]}"
        }
        for row in rows
    ]

def get_all_persons_embedding():
    """Retrieve all person embeddings from the database."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()
    cursor.execute("SELECT person_id, face_embedding FROM Person")
    rows = cursor.fetchall()
    conn.close()
    return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

def log_entry(person_id, captured_face_path):
    """Log an entry for a recognized person."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()
    in_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO EntryLog (person_id, captured_face, location, in_time) VALUES (?, ?, ?, ?)", 
                   (person_id, captured_face_path, "Location Info", in_time))
    conn.commit()
    conn.close()


def delete_log(log_id):
    """Delete a log from the database."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM EntryLog WHERE serial_no = ?", (log_id,))
    conn.commit()
    conn.close()

def delete_person_and_logs(person_id):
    """Delete a person and all associated logs."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM EntryLog WHERE person_id = ?", (person_id,))
    cursor.execute("DELETE FROM Person WHERE person_id = ?", (person_id,))
    conn.commit()
    conn.close()

def get_logs_for_person(person_id):
    """Retrieve logs for a specific person."""
    conn = sqlite3.connect("surveillance81_2.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT serial_no, in_time, captured_face, location
        FROM EntryLog
        WHERE person_id = ?
        ORDER BY in_time DESC
    """, (person_id,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "serial_no": row[0],
            "in_time": row[1],
            "captured_face": f"/captured_faces/{row[2]}",
            "location": row[3]
        }
        for row in rows
    ]
