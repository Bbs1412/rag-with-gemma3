"""
# Files.py
- This module handles file uploads, storage, and management of user files
- This deals with disk storage of files.
"""

import os
import time
import shutil
import base64
import fitz  # PyMuPDF
from io import BytesIO
from typing import Tuple

from logger import get_logger
log = get_logger(name="FILES", log_to_console=False)

UPLOADS_PATH = os.path.abspath("./user_uploads/")


def check_create_uploads_folder() -> str:
    """Check and create the uploads directory if it doesn't exist."""

    if not os.path.exists(UPLOADS_PATH):
        os.makedirs(UPLOADS_PATH)
        log.info(f"Uploads folder created at: {UPLOADS_PATH}")
    else:
        log.info(f"Uploads folder already exists at: {UPLOADS_PATH}")
    return UPLOADS_PATH


def create_user_uploads_folder(user_id: str) -> bool:
    """Create a user-specific uploads directory if it doesn't exist."""

    try:
        user_upload_path = os.path.join(UPLOADS_PATH, user_id)
        if not os.path.exists(user_upload_path):
            os.makedirs(user_upload_path)
            log.info(f"Uploads folder created for user {user_id} at: {user_upload_path}")
        else:
            log.info(f"Uploads folder already exists for user {user_id} at: {user_upload_path}")
        return True

    except Exception as e:
        log.error(f"Error creating uploads folder for user {user_id}: {repr(e)}")
        return False


def delete_file(user_id: str, file_name: str) -> bool:
    """Delete a user file from the uploads directory.
    Args:
        user_id (str): The name of the user whose file is to be deleted.
        file_name (str): The name of the file to be deleted.
    Returns:
        bool: True if the file was successfully deleted, False otherwise.
    """

    file_path = os.path.join(UPLOADS_PATH, user_id, file_name)
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            log.info(f"User {user_id} - File deleted: {file_name}")
            return True
        except Exception as e:
            log.error(f"User {user_id} - Error deleting file {file_name}: {repr(e)}")
            return False
    else:
        log.warning(f"User {user_id} - File not found for deletion: {file_name}")
        return False


def save_file(user_id: str, file_value_binary: bytes, file_name: str) -> Tuple[bool, str]:
    """Save the streamlit uploaded user file to the uploads directory.
    Args:
        user_id (str): The name of the user uploading the file.
        file_value_binary (bytes): The binary content of the uploaded file.
        file_name (str): The original name of the uploaded file.
    Returns:
        Tuple[bool, str]: A tuple containing a success flag and the saved file name or failure message.
    """

    try:
        base_name = file_name[:file_name.rfind(".")]
        ext = file_name[file_name.rfind("."):]
        new_file_name = base_name.replace(" ", "_").replace(".", "_") + ext

        # check if same name already exists:
        user_upload_path = os.path.join(UPLOADS_PATH, user_id)
        while os.path.exists(os.path.join(user_upload_path, new_file_name)):
            new_file_name = "n_" + new_file_name

        # Save the file
        file_path = os.path.join(user_upload_path, new_file_name)

        with open(file_path, "wb") as f:
            f.write(file_value_binary)
            log.info(f"User {user_id} - File saved: {new_file_name}")

        return True, new_file_name

    except Exception as e:
        log.error(f"User {user_id} - Error saving file {file_name}: {repr(e)}")
        return False, "Error saving file!"


def get_pdf_iframe(user_id: str, file_name: str, num_pages: int = 5) -> Tuple[bool, str]:
    """Return first n pages of asked pdf in an iframe.
    
    Args:
        user_id (str): The name of the user whose file is to be loaded.
        file_name (str): The name of the PDF file to be loaded.
        num_pages (int): The number of pages to include in the iframe.
    Returns:
        Tuple[bool, str]: A tuple containing a success flag and the HTML iframe string or an error message.
    """
    log.info(f"Loading PDF: {user_id}/{file_name}")
    abs_path = os.path.join(UPLOADS_PATH, user_id, file_name)
    print(f"Absolute path: {abs_path}")

    if not os.path.isfile(abs_path):
        log.error(f"File not found at {abs_path}!")
        return False, "File not found!"

    try:
        # Open original PDF
        doc = fitz.open(abs_path)

        # Create new PDF in memory
        new_pdf = fitz.open()
        for i in range(min(num_pages, len(doc))):
            new_pdf.insert_pdf(doc, from_page=i, to_page=i)

        # Write to in-memory buffer
        pdf_buffer = BytesIO()
        new_pdf.save(pdf_buffer)
        new_pdf.close()
        doc.close()

        # Base64 encode the trimmed version
        base64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode("utf-8")

        # Close the buffer
        pdf_buffer.close()
        log.info(f"PDF iframe loaded successfully: {user_id}/{file_name}")

        return True, f"""
            <iframe
                src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=1&page=1&view=FitH"
                width=100%
                height=300rem
                type="application/pdf"
            ></iframe>
        """

    except Exception as e:
        log.error(f"Error loading PDF {file_name} for user {user_id}: {repr(e)}")
        return False, "Some error occurred while loading the PDF!"


if __name__ == "__main__":
    user = "test_user"

    # Example usage
    check_create_uploads_folder()

    # Create a user uploads folder
    create_user_uploads_folder(user)

    # Save a sample file
    sample_file_content = b"This is a sample file content."
    print(save_file(user, sample_file_content, "sample.a.b c.d.pdf"))

    # Get PDF iframe:
    file = input(f"Paste one file here, `{UPLOADS_PATH}/{user}/` and paste just the file name: ")
    print(get_pdf_iframe(user, file, num_pages=1))


