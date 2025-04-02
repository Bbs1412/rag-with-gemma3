import streamlit as st

import os
import time
import shutil
import random
import string
from typing import Optional, List, Literal

from log import Logger
from llm import get_response, get_response_stream

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
    # Check uploads folder:
    uploads_path = f"uploads"
    if not os.path.exists(uploads_path):
        os.makedirs(uploads_path)

    # Initialize Logger:
    st.session_state.logger = Logger(name="Streamlit")
    logger = st.session_state.logger

    # Assign session id:
    def generate_random(length=10):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    session_id = generate_random()
    while session_id in os.listdir("./uploads"):
        session_id = generate_random()
    logger.log(f"New Session ID: {session_id}", level='info')

    # Create session folder:
    os.makedirs(name=f"uploads/{session_id}")
    st.session_state.session_id = session_id
    st.session_state.user_uploads = f"{uploads_path}/{st.session_state.session_id}"

    # Delete old folders: (< 1 day old)
    current_time = time.time()
    num_days = 1
    cutoff_time = current_time - (num_days * 24 * 60 * 60)

    for folder_name in os.listdir(uploads_path):
        folder_path = os.path.join(uploads_path, folder_name)
        if os.path.isdir(folder_path):
            folder_creation_time = os.path.getctime(folder_path)

            if folder_creation_time < cutoff_time:
                shutil.rmtree(folder_path)
                logger.log(f"Deleted folder: {folder_name}", level='warning')

    # Initialize messages:
    st.session_state.chat_history = [
        Message('assistant', "How may I help you today?"),
        # Message("human", "Help me in some thing...")
    ]

    # Set flag to true:
    st.session_state.initialized = True


# All variables in session state:
session_id, user_uploads, logger, chat_history, initialized = (
    st.session_state.session_id,
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


# ------------------------------------------------------------------------------
# Page content:
# ------------------------------------------------------------------------------

# selectbox to choose one among all the uploaded files
# preview of all the uploaded files
# st.sidebar


a, b = st.columns([0.5, 9.5], vertical_alignment='bottom', gap='small')
a.image("./assets/gemma.jpg")
b.header(":green[RAG] with :blue[Gemma-3]", divider='rainbow')

# if "first" not in st.session_state:
# st.session_state['first'] = True
for message in st.session_state.chat_history:
    if message.type == 'human':
        write_as_human(message.content, message.filenames)

    elif message.type == 'assistant':
        write_as_ai(message.content)

    # st.json(message.__dict__)


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

    # Save and write it to the chat:
    st.session_state.chat_history.append(new_message)
    write_as_human(new_message.content, new_message.filenames)

    # Get response and write it:
    full = ""

    with st.chat_message(name='assistant', avatar='assistant'):
        with st.spinner("Generating response..."):
            container = st.empty()

            resp = get_response_stream(new_message.content, dummy=False)
            for chunk in resp:
                full += chunk
                container.container(border=True).markdown(full + "|")

    st.session_state.chat_history.append(Message("assistant", full))
    st.rerun()

    
