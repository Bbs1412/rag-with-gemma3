from log import Logger

import os
import time
import shutil
import base64
import fitz  # PyMuPDF
from io import BytesIO
from typing import Tuple

logger = Logger(name="FILES", log_to_console=False)
UPLOADS_PATH = os.path.abspath("./uploads/")


def check_create_uploads_folder() -> str:
    """Check and create the uploads directory if it doesn't exist."""
    if not os.path.exists(UPLOADS_PATH):
        os.makedirs(UPLOADS_PATH)
        logger.log(f"Uploads folder created at: {UPLOADS_PATH}", level='info')
    else:
        logger.log(
            f"Uploads folder already exists at: {UPLOADS_PATH}", level='info')
    return UPLOADS_PATH


def delete_old_files(time_in_hours: int = 24) -> None:
    """Delete files older than the specified time in hours."""
    current_time = time.time()
    cutoff_time = current_time - (time_in_hours * 3600)

    for file in os.listdir(UPLOADS_PATH):
        full_path = os.path.join(UPLOADS_PATH, file)
        creation_time = os.path.getctime(full_path)

        if creation_time < cutoff_time:
            shutil.rmtree(full_path)
            logger.info(f"Deleted old file: {file}")


def save_file(name: str, file_value_binary: bytes) -> Tuple[bool, str]:
    """Save the uploaded file to the specified directory."""

    try:
        new_file_name = f"{"_".join(name.split(".")[:-1])}.{name.split(".")[-1]}"

        with open(os.path.join('uploads', new_file_name), "wb") as f:
            f.write(file_value_binary)
            logger.log(f"File saved: {new_file_name}", level='info')
        return True, new_file_name

    except Exception as e:
        logger.log(f"Error saving file: {e}", level='error')
        return False, "Error saving file!"

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


def get_pdf_iframe(abs_path: str) -> Tuple[bool, str]:
    """Return first 10 pages of asked pdf in an iframe."""
    logger.log(f"Loading PDF: {abs_path}", level='info')

    if not os.path.isfile(abs_path):
        logger.log(f"File not found at {abs_path}!", level='error')
        return False, "File not found!"

    try:
        # Open original PDF
        doc = fitz.open(abs_path)

        # Create new PDF in memory
        new_pdf = fitz.open()
        for i in range(min(5, len(doc))):
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

        logger.log(f"PDF trimmed and encoded successfully.", level='info')

        return True, f"""
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=1&page=1&view=FitH" 
                width=100%
                height=300rem
                type="application/pdf"
            ></iframe>
        """

    except Exception as e:
        # st.error(f"Error loading PDF: {e}")
        logger.log(f"Error loading PDF: {e}", level='error')
        return False, "Some error occurred while loading the PDF!"
