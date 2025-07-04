"""
Module for managing SQLite database operations.
- This module provides functions to track user uploaded / generated files / data.
- It includes creating tables, adding files and embeddings, and deleting the database.
- Each table has col 'available' to mark if the record is still valid or has been deleted.
"""

import bcrypt
import os
import sqlite3
from typing import List
from pathlib import Path
from logger import get_logger

import pytz
from datetime import datetime, timedelta

log = get_logger(name="sq_db")
DB_PATH = Path("user_data.db")
IST = pytz.timezone('Asia/Kolkata')


# ------------------------------------------------------------------------------
# Database Management Functions:
# ------------------------------------------------------------------------------

def get_connection():
    """Creates and returns a SQLite database connection.
    - The connection is set to allow multiple threads to access it.
    - The database file is created if it does not exist.
    """

    return sqlite3.connect(DB_PATH, check_same_thread=False)


def delete_database() -> bool:
    """Deletes the SQLite database file if it exists."""

    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            log.info(f"Database file '{DB_PATH}' deleted successfully.")
            return True
        except OSError as e:
            log.error(f"Error deleting database file '{DB_PATH}': {e}")
            return False
    else:
        log.warning(f"Database file '{DB_PATH}' does not exist, nothing to delete.")
        return True


def create_tables():
    """Creates the necessary tables in the SQLite database.
    - The `uploads` table tracks user uploaded files.
    - The `embeddings` table tracks embeddings associated with those files' chunks.
    """

    with get_connection() as conn:
        cur = conn.cursor()

        # USERS(user_id*, name, password_hash, last_login)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                last_login TEXT
            )
        """)

        # UPLOADS(file_id*, user_id^, filename, created_at, available)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                available INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
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


# ------------------------------------------------------------------------------
# User Management Functions:
# ------------------------------------------------------------------------------

def add_user(user_id: str, name: str, password: str) -> bool:
    """Adds a new user to the database.

    Args:
        user_id (str): The unique ID of the user.
        name (str): The name of the user.
        password (str): The raw password of the user.

    Returns:
        bool: True if the user was added successfully, False otherwise.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            ist_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            cur.execute("""
                INSERT INTO users (user_id, name, password_hash, last_login)
                VALUES (?, ?, ?, ?)
            """, (user_id, name, hashed_password.decode('utf-8'), ist_time))
            conn.commit()

            if cur.rowcount == 0:
                log.error(f"Failed to insert user '{name}' with ID '{user_id}'")
                return False

            log.info(f"User '{name}' added with ID '{user_id}'")
            return True
    except sqlite3.Error as e:
        log.error(f"SQLite error while adding user '{name}' with ID '{user_id}': {e}")
        return False


def check_user_exists(user_id: str) -> bool:
    """Checks if a user exists in the database.

    Args:
        user_id (str): The ID of the user to check.

    Returns:
        bool: True if the user exists, False otherwise.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            result = cur.fetchone()
            if result:
                log.info(f"User '{user_id}' exists in the database.")
                return True
            else:
                log.warning(f"User '{user_id}' does not exist in the database.")
                return False
    except sqlite3.Error as e:
        log.error(f"SQLite error while checking user '{user_id}': {e}")
        return False


def authenticate_user(user_id: str, password: str) -> tuple[bool, str]:
    """Authenticates a user by checking their ID and password hash.
    + If the user exists and the password matches, it updates the last login time.

    Args:
        user_id (str): The ID of the user to authenticate.
        password (str): The raw password of the user.

    Returns:
        tuple (bool, str):
            + bool: True if the authentication is successful, False otherwise.
            + str: A message indicating the error or The user-name if authentication is successful.
    """
    try:
        with get_connection() as conn:
            # Fetch the hashed password from the database
            cur = conn.cursor()
            cur.execute("SELECT password_hash, name FROM users WHERE user_id = ?", (user_id,))
            result = cur.fetchone()

            if result:
                hashed_password = result[0].encode('utf-8')

                # Check if the provided password matches the hashed password
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                    log.info(f"User '{user_id}' authenticated successfully.")
                    # Update last login time
                    ist_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

                    cur.execute("""
                        UPDATE users SET last_login = ?
                        WHERE user_id = ?
                    """, (ist_time, user_id))
                    conn.commit()
                    log.info(f"Authentication successful for '{user_id}' @ {ist_time}")
                    return True, result[1]  # Return True and the user's name

                else:
                    log.warning(f"Authentication failed for user '{user_id}': Incorrect password.")
                    return False, "Incorrect password."

            else:
                log.warning(f"Authentication failed for user '{user_id}': User does not exist.")
                return False, "User does not exist."

    except sqlite3.Error as e:
        log.error(f"SQLite error while authenticating user '{user_id}': {e}")
        return False, "Database error during authentication."


# ------------------------------------------------------------------------------
# File Management Functions:
# ------------------------------------------------------------------------------

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


def get_user_files(user_id: str) -> List[str]:
    """Retrieves all files uploaded by a user.
    - Only retrieves files that are marked as available (not deleted).

    Args:
        user_id (str): The ID of the user whose files are being queried.

    Returns:
        List[str]: A list of file names uploaded by the user.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT filename FROM uploads
                WHERE user_id = ? AND available = 1
            """, (user_id,))
            files = cur.fetchall()

            log.info(f"Retrieved {len(files)} files for user '{user_id}'")
            return [file[0] for file in files]

    except sqlite3.Error as e:
        log.error(f"SQLite error while retrieving files for user '{user_id}': {e}")
        return []


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
                log.info(f"File '{file_name}' resolved ID={result[0]} for user '{user_id}'")
                return result[0]
            else:
                log.warning(f"File '{file_name}' not found to resolve ID for user '{user_id}'")
                return -1

    except sqlite3.Error as e:
        log.error(
            f"SQLite error while retrieving file ID for '{file_name}' and user '{user_id}': {e}")
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


# ------------------------------------------------------------------------------
# Embedding Management Functions:
# ------------------------------------------------------------------------------

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


def mark_embeddings_removed(vector_ids: List[str]) -> bool:
    """Marks an embedding as unavailable (deleted) in the database.

    Args:
        vector_ids (List[str]): A list of vector IDs to be marked as unavailable.
    Returns:
        bool: True if the embeddings were marked as unavailable, False otherwise.
    """
    if not vector_ids:
        log.warning("No vector IDs provided to mark as removed.")
        return False

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            placeholders = ', '.join('?' for _ in vector_ids)
            cur.execute(f"""
                UPDATE embeddings SET available = 0
                WHERE vector_id IN ({placeholders})
            """, vector_ids)
            conn.commit()

            if cur.rowcount != len(vector_ids):
                log.warning(f"Marked {cur.rowcount}/{len(vector_ids)} embeddings as removed.")
                return False
            else:
                log.info(f"Marked all ({len(vector_ids)}) embeddings as removed.")
                return True

    except sqlite3.Error as e:
        log.error(f"SQLite error while marking embeddings as removed: {e}")
        return False


# ------------------------------------------------------------------------------
# Main Execution Block for Testing:
# ------------------------------------------------------------------------------

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

    # Check USERs management:
    print("\nChecking user management:")
    assert add_user("abc", "a b c", "pass") == True, "User addition failed"
    assert check_user_exists("abc") == True, "User existence check failed"
    assert authenticate_user("abc", "pass") == (True, 'a b c'), "User authentication failed"
    print("\t - User management checks passed successfully.")

    # Cleanup:
    print("\nAll tests passed successfully!")
    print("\t - Cleaning up the database...")
    delete_database()
    print("\t - Database cleaned up.")
    create_tables()
    print("\t - Blank Tables recreated successfully.")
