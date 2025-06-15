"""
Module for managing SQLite database operations.
- This module provides functions to track user uploaded / generated files / data.
- It includes creating tables, adding files and embeddings, and deleting the database.
- Each table has col 'available' to mark if the record is still valid or has been deleted.
"""

import sqlite3
from typing import List
from pathlib import Path
from logger import get_logger

import pytz
from datetime import datetime, timedelta

log = get_logger(name="sq_db")
DB_PATH = Path("user_data.db")
IST = pytz.timezone('Asia/Kolkata')


def get_connection():
    """Creates and returns a SQLite database connection.
    - The connection is set to allow multiple threads to access it.
    - The database file is created if it does not exist.
    """

    return sqlite3.connect(DB_PATH, check_same_thread=False)


def delete_database() -> bool:
    """Deletes the SQLite database file if it exists."""

    if DB_PATH.exists():
        DB_PATH.unlink()
        return True
    else:
        raise FileNotFoundError(f"Database file '{DB_PATH}' does not exist.")


def create_tables():
    """Creates the necessary tables in the SQLite database.
    - The `uploads` table tracks user uploaded files.
    - The `embeddings` table tracks embeddings associated with those files' chunks.
    """

    with get_connection() as conn:
        cur = conn.cursor()

        # UPLOADS(file_id*, user_id, filename, created_at, available)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                available INTEGER DEFAULT 1
            )
        """)

        # EMBEDDINGS(file_id^, vector_id, available)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                vector_id TEXT NOT NULL,
                available INTEGER DEFAULT 1,
                FOREIGN KEY (file_id) REFERENCES uploads(file_id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        log.info("Database tables created successfully.")


def add_file(user_id: str, filename: str) -> int:
    """Adds a file upload record to the database.

    Args:
        user_id (str): The ID of the user uploading the file.
        filename (str): The name of the file being uploaded.
    Returns:
        int: The ID (=file_id) of the newly created file record, or -1 if an error occurred.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            ist_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

            cur.execute(
                "INSERT INTO uploads (user_id, filename, created_at) VALUES (?, ?, ?)",
                (user_id, filename, ist_time)
            )
            conn.commit()

            if cur.lastrowid is None:
                log.error(f"Failed to insert file '{filename}' for '{user_id}', no ID was returned")
                return -1

            log.info(f"File '{filename}' added for user '{user_id}' with ID {cur.lastrowid}")
            return cur.lastrowid

    except sqlite3.Error as e:
        log.error(f"SQLite error while adding file '{filename}' for user '{user_id}': {e}")
        return -1


def add_embedding(file_id: int, vector_id: str) -> bool:
    """Adds an embedding record for a file in the database.

    Args:
        file_id (int): The ID of the file to which the embedding belongs.
        vector_id (str): The ID of the embedding vector.

    Returns:
        bool: True if the embedding was added successfully, False otherwise.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO embeddings (file_id, vector_id) VALUES (?, ?)",
                (file_id, vector_id,)
            )
            conn.commit()
            log.info(f"Embedding added for file ID {file_id} with vector ID '{vector_id}'")
            return True

    except sqlite3.Error as e:
        log.error(f"SQLite error while adding embedding for file ID {file_id}: {e}")
        return False


def get_old_files(user_id: str, time: int = 12*3600) -> dict[str, List[str]]:
    """Retrieves files uploaded by a user that are older than a specified time.

    Args:
        user_id (str): The ID of the user whose files are being queried.
        time (int): The time in seconds to consider a file as old. Default is 12 hours.

    Returns:
        dict[str, List[str]]: A dictionary with keys [files, embeddings] and values as lists of file names and embedding vector IDs respectively.    
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()

            # Calc cutoff time as string in IST
            cutoff_time = datetime.now(IST) - timedelta(seconds=time)
            threshold_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")

            # Get old files
            cur.execute("""
                SELECT filename, file_id FROM uploads
                WHERE user_id = ? AND available = 1 AND created_at < ?
            """, (user_id, threshold_str))
            files = cur.fetchall()

            # Get embeddings for those files
            file_ids = [file[1] for file in files]
            if not file_ids:
                return {"files": [], "embeddings": []}

            placeholders = ', '.join('?' for _ in file_ids)
            cur.execute(f"""
                SELECT vector_id FROM embeddings
                WHERE file_id IN ({placeholders}) AND available = 1
            """, file_ids)
            embeddings = cur.fetchall()

            log.info(
                f"Retrieved {len(files)} old files and {len(embeddings)} embeddings for user '{user_id}'")
            return {
                "files": [file[0] for file in files],
                "embeddings": [embedding[0] for embedding in embeddings]
            }

    except sqlite3.Error as e:
        log.error(f"SQLite error while retrieving old files for user '{user_id}': {e}")
        return {"files": [], "embeddings": []}


def get_file_id_by_name(user_id: str, file_name: str) -> int:
    """Retrieves the file ID for a given file name and user ID.
    - Checks only the active files and not old or deleted files.

    Args:
        user_id (str): The ID of the user who owns the file.
        file_name (str): The name of the file to search for.

    Returns:
        int: The ID of the file if found, -1 otherwise.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT file_id FROM uploads
                WHERE user_id = ? AND filename = ? AND available = 1
            """, (user_id, file_name))
            result = cur.fetchone()

            if result:
                log.info(f"File '{file_name}' found for user '{user_id}' with ID {result[0]}")
                return result[0]
            else:
                log.warning(f"File '{file_name}' not found for user '{user_id}'")
                return -1

    except sqlite3.Error as e:
        log.error(f"SQLite error while retrieving file ID for '{file_name}' and user '{user_id}': {e}")
        return -1

def mark_file_removed(user_id: str, file_id: int) -> bool:
    """Marks a file as unavailable (deleted) in the database.

    Args:
        user_id (str): The ID of the user who owns the file.
        file_id (int): The ID of the file to be marked as unavailable.

    Returns:
        bool: True if the file was marked as unavailable, False otherwise.

    ## `Warning:`
        - This function does not physically delete any file or entry
        - It is just useful for managing / tracking user data
        - Use this to:
            1. Get the embedding doc ids and then use them to delete in FAISS
            2. Get file names and delete files using files module
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE uploads SET available = 0
                WHERE user_id = ? AND file_id = ?
            """, (user_id, file_id))
            conn.commit()

            if cur.rowcount == 0:
                log.warning(f"No file found for user '{user_id}' with ID {file_id}")
                return False

            log.info(f"File ID {file_id} marked as deleted for user '{user_id}'")
            return True

    except sqlite3.Error as e:
        log.error(f"SQLite error while removing file ID {file_id} for user '{user_id}': {e}")
        return False


if __name__ == "__main__":
    # Clear the database and reCreate:
    print("Setup:")
    delete_database()
    print("\t - Database cleared.")
    create_tables()
    print("\t - Database and tables created successfully.")

    # Add files:
    print("\nAdding files and embeddings:")
    f1 = add_file(user_id="test_user_1", filename="user1_file1.txt")
    print(f"\t - File added: {f1}")
    f2 = add_file(user_id="test_user_2", filename="user2_file1.txt")
    f3 = add_file(user_id="test_user_1", filename="user1_file2.txt")

    # Add embeddings for the files
    # user_1
    add_embedding(file_id=f1, vector_id="user1_f1_e1")
    add_embedding(file_id=f3, vector_id="user1_f2_e1")
    # user_2
    add_embedding(file_id=f2, vector_id="user2_f1_e1")
    add_embedding(file_id=f2, vector_id="user2_f1_e2")
    print(f"\t - Embeddings added for files: {f1}, {f2}, {f3}")

    # Wait for a while to simulate old files
    from time import sleep
    print("\nWait a sec...", end="\r")
    sleep(2)

    # Retrieve old files for the user
    print(f"Retrieve Old files:")
    print("User 1:")
    old1 = get_old_files(user_id="test_user_1", time=1)
    for k, v in old1.items():
        print(f"\t{k}: {v}")

    print("User 2:")
    old2 = get_old_files(user_id="test_user_2", time=1)
    for k, v in old2.items():
        print(f"\t{k}: {v}")

    # Tests:

    # Addition:
    assert add_embedding(file_id=f1, vector_id="user1_f1_e1") == True, "Embedding addition failed"
    # Old Retrieval:
    assert old1["files"] == ["user1_file1.txt", "user1_file2.txt"], "User 1 old files mismatch"
    assert old1["embeddings"] == ["user1_f1_e1", "user1_f2_e1"], "User 1 old embeddings mismatch"
    assert old2["files"] == ["user2_file1.txt"], "User 2 old files mismatch"
    assert old2["embeddings"] == ["user2_f1_e1", "user2_f1_e2"], "User 2 old embeddings mismatch"

    # Removal:
    print("\nRemoving files:")
    assert mark_file_removed("test_user_2", f2) == True, "File removal failed for user 1"
    print("\t - File remove successful for user 2")
    
    # Check cascade delete:
    emb = get_old_files(user_id="test_user_2", time=1)["embeddings"]
    assert emb == [], "Embeddings not deleted after file removal for user 2"
    print("\t - Embeddings deleted after file removal for user 2")

    print("\nAll tests passed successfully!")

    # Cleanup:
    print("\t - Cleaning up the database...")
    delete_database()
    print("\t - Database cleaned up.")
    create_tables()
    print("\t - Blank Tables recreated successfully.")
