import streamlit as st

import os
import time
import shutil
from typing import Optional, List, Literal

from log import Logger
from llm import get_response, get_response_stream
from files import check_create_uploads_folder, delete_old_files, save_file, get_pdf_iframe

# ------------------------------------------------------------------------------
# Page Config:
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG with Gemma3",
    page_icon="✨",
    layout='wide',
    initial_sidebar_state='expanded',
)


# ------------------------------------------------------------------------------
# Page consistent settings and initializations:
# ------------------------------------------------------------------------------

class Message:
    type: Literal['assistant', 'human']
    content: str
    filenames: Optional[List[str]]

    def __init__(
        self, type: Literal['assistant', 'human'],
        content: str, filenames: Optional[List[str]] = None
    ):
        self.type = type
        self.content = content
        self.filenames = filenames


if "initialized" not in st.session_state:
    st.session_state.uploads_path = check_create_uploads_folder()
    # delete_old_files(time_in_hours=24)

    # Initialize Logger:
    st.session_state.logger = Logger(name="Streamlit")
    logger = st.session_state.logger
    logger.log("Streamlit initialized.", level='info')

    # Initialize messages:
    st.session_state.chat_history = [
        Message('assistant', "How may I help you today?"),
        # Message("human", "Help me in some thing...")
    ]

    # User uploads:
    st.session_state.user_uploads = []

    # Set flag to true:
    st.session_state.initialized = True


# All variables in session state:
uploads_path, user_uploads, logger, chat_history, initialized = (
    st.session_state.uploads_path,
    st.session_state.user_uploads,
    st.session_state.logger,
    st.session_state.chat_history,
    st.session_state.initialized
)


# ------------------------------------------------------------------------------
# Helper functions:
# ------------------------------------------------------------------------------


def write_as_ai(text):
    with st.chat_message(name='assistant', avatar='assistant'):
        st.markdown(text)


def write_as_human(text: str, filenames: Optional[List[str]] = None):
    with st.chat_message(name='user', avatar='user'):
        st.markdown(text)
        if filenames:
            files = ", ".join([f"`'{file}'`" for file in filenames])
            st.caption(f"🔗 Attached file(s): {files}.")


def handle_uploaded_files(uploaded_files) -> bool:
    progress_status = ""

    with st.chat_message(name='assistant', avatar='./assets/settings_3.png'):
        with st.spinner("Processing files..."):
            container = st.empty()

            def write_progress(msg, in_progress=False):
                nonlocal progress_status
                if in_progress:
                    curr = progress_status + f"- ⏳ {msg}"
                else:
                    progress_status += f"- ✅ {msg}\n"
                    curr = progress_status

                container.container(border=True).markdown(curr)

            try:
                for i, file in enumerate(uploaded_files):
                    progress_status += f"\n📂 Processing file {i+1} of {len(uploaded_files)}...\n"

                    # Save the uploaded file to the uploads folder:
                    write_progress("Saving file...", True)
                    status, file_name = save_file(file.name, file.getvalue())
                    time.sleep(st.secrets.llm.per_step_delay)
                    write_progress(f"File `{file_name}` saved successfully.")

                    # Embed the file:
                    write_progress("Embedding file...", True)
                    time.sleep(st.secrets.llm.per_step_delay)
                    write_progress("Generated embeddings successfully...")

                    # Add file in session_state:
                    st.session_state.user_uploads.append(file_name)
                    time.sleep(st.secrets.llm.end_delay)

                    # Done:
                    time.sleep(st.secrets.llm.end_delay)
                    logger.log(
                        f"File `{file_name}` processed successfully.", level='info')

                return True

            except Exception as e:
                logger.log(f"Error processing files: {e}", level='error')
                return False


# ------------------------------------------------------------------------------
# Sidebar:
# ------------------------------------------------------------------------------


st.sidebar.subheader("📂 Files")

selected_file = st.sidebar.selectbox(
    label="Choose File to ***Preview***",
    index=0,
    options=st.session_state.user_uploads,
)

if selected_file:
    file = os.path.join(uploads_path, selected_file)
    status, content = get_pdf_iframe(file)

    if status:
        st.sidebar.markdown(content, unsafe_allow_html=True)
    else:
        st.sidebar.error(content, icon="😕")


# ------------------------------------------------------------------------------
# Page content:
# ------------------------------------------------------------------------------

a, b = st.columns([0.65, 9.35], vertical_alignment='bottom', gap='small')
a.image("./assets/gemma.jpg", use_container_width=True)
b.header(":green[RAG] with :blue[Gemma-3]", divider='rainbow')

for message in st.session_state.chat_history:
    if message.type == 'human':
        write_as_human(message.content, message.filenames)

    elif message.type == 'assistant':
        write_as_ai(message.content)


if user_message := st.chat_input(
    placeholder="Enter any queries here...",
    max_chars=1000,
    accept_file='multiple',
    file_type=['pdf'],
    # on_submit=submit_handler
):
    # Create Message object from the user input:
    new_message = Message(
        type="human",
        content=user_message.text,
        filenames=[
            file.name for file in user_message.files] if user_message.files else None
    )

    # Save it to the chat:
    st.session_state.chat_history.append(new_message)
    # For now, write it on screen:
    write_as_human(new_message.content, new_message.filenames)

    # Handle the files if any:
    if user_message.files:
        if handle_uploaded_files(user_message.files):
            st.toast("Files processed successfully!", icon="✅")
        else:
            st.error("Error processing files. Please try again.", icon="🚫")

    # Get response and write it:
    full = ""
    with st.chat_message(name='assistant', avatar='assistant'):
        with st.spinner("Generating response..."):
            container = st.empty()

            resp = get_response_stream(new_message.content, dummy=True)
            for chunk in resp:
                full += chunk
                container.container(border=True).markdown(full + "|")

    st.session_state.chat_history.append(Message("assistant", full))
    st.rerun()
