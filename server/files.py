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


if __name__ == "__main__":
    # Example usage
    check_create_uploads_folder()

    # Save a sample file
    sample_file_content = b"This is a sample file content."
    print(
        save_file("test_user", sample_file_content, "sample.a.b c.d.txt")
    )

# ------------------------------------------------------------------------------
# Old Code Snippets
# ------------------------------------------------------------------------------

# def delete_old_files(time_in_hours: int = 24) -> None:
#     """Delete files older than the specified time in hours."""
#     current_time = time.time()
#     cutoff_time = current_time - (time_in_hours * 3600)
#     old_files = []

#     for file in os.listdir(UPLOADS_PATH):
#         full_path = os.path.join(UPLOADS_PATH, file)
#         creation_time = os.path.getctime(full_path)

#         if creation_time < cutoff_time:
#             old_files.append(full_path)

#     if old_files:
#         deleted_files, failed_files = [], []
#         for file in old_files:
#             try:
#                 os.remove(file)
#                 deleted_files.append(file)
#             except Exception as e:
#                 failed_files.append(file + f" (Error: {repr(e)})")

#         if deleted_files:
#             log.info(f"Deleted files: [{', '.join(deleted_files)}]")
#         if failed_files:
#             log.error(f"Failed to delete files: {'; '.join(failed_files)}")


# def save_file(name: str, file_value_binary: bytes) -> Tuple[bool, str]:
#     """Save the uploaded file to the specified directory."""

#     try:
#         new_file_name = f"{"_".join(name.split(".")[:-1])}.{name.split(".")[-1]}"

#         with open(os.path.join('uploads', new_file_name), "wb") as f:
#             f.write(file_value_binary)
#             log.info(f"File saved: {new_file_name}")
#         return True, new_file_name

#     except Exception as e:
#         log.error(f"Error saving file: {e}")
#         return False, "Error saving file!"


# def get_pdf_iframe(abs_path: str):
#     if not os.path.isfile(abs_path):
#         st.error(f"File not found: {abs_path}")
#         return

#     with open(abs_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     return f"""
#     <iframe
#         src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=1&page=1&view=FitH"
#         width=100%
#         height=300rem
#         type="application/pdf"
#     ></iframe>
#     """


# def get_pdf_iframe(abs_path: str) -> Tuple[bool, str]:
#     """Return first 10 pages of asked pdf in an iframe."""
#     logger.log(f"Loading PDF: {abs_path}", level='info')

#     if not os.path.isfile(abs_path):
#         logger.log(f"File not found at {abs_path}!", level='error')
#         return False, "File not found!"

#     try:
#         # Open original PDF
#         doc = fitz.open(abs_path)

#         # Create new PDF in memory
#         new_pdf = fitz.open()
#         for i in range(min(5, len(doc))):
#             new_pdf.insert_pdf(doc, from_page=i, to_page=i)

#         # Write to in-memory buffer
#         pdf_buffer = BytesIO()
#         new_pdf.save(pdf_buffer)
#         new_pdf.close()
#         doc.close()

#         # Base64 encode the trimmed version
#         base64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode("utf-8")

#         # Close the buffer
#         pdf_buffer.close()

#         logger.log(f"PDF trimmed and encoded successfully.", level='info')

#         return True, f"""
#             <iframe
#                 src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=1&page=1&view=FitH"
#                 width=100%
#                 height=300rem
#                 type="application/pdf"
#             ></iframe>
#         """

#     except Exception as e:
#         # st.error(f"Error loading PDF: {e}")
#         logger.log(f"Error loading PDF: {e}", level='error')
#         return False, "Some error occurred while loading the PDF!"
