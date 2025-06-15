import requests
import streamlit as st

import os
import json
import time
from contextlib import contextmanager
from typing import Optional, List, Literal

import server.logger as logger
# from .server import logger

# from llm import get_response, get_response_stream, get_conversational_rag_response
# from files import check_create_uploads_folder, delete_old_files, save_file, get_pdf_iframe
from files import get_pdf_iframe

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
    # List of filenames attached to the message
    # These names will be original file names, might be diff than actual saved on server
    # Hence, Chat-UI and sidebar might show same file with different names.

    def __init__(
        self, type: Literal['assistant', 'human'],
        content: str, filenames: Optional[List[str]] = None
    ):
        self.type = type
        self.content = content
        self.filenames = filenames


if "initialized" not in st.session_state:
    # Initialize Logger:
    st.session_state.logger = logger.get_logger(name="Streamlit")
    log = st.session_state.logger
    log.info("Streamlit initialized.")

    # Initialize the session with server::
    st.session_state.server_ip = "http://127.0.0.1:8000"
    try:
        resp = requests.post(
            f"{st.session_state.server_ip}/login",
            json={"login_id": "dummy", "password": "dummy"}
        )

        if resp.status_code == 200:
            session_id = resp.json().get("user_id")
            st.session_state.session_id = session_id
            log.info(f"Server session initialized successfully. Session ID: {session_id}")
        else:
            st.session_state.session_id = None
            log.error(f"Failed to initialize server session: {resp.text}")
            raise Exception("Server did not respond as expected.")

    except requests.RequestException as e:
        log.error(f"Error initializing server session: {e}")
        st.error(
            "Failed to connect to the server. Please check your connection or server status.",
            icon="🚫"
        )
        st.stop()

    # Initialize messages:
    st.session_state.chat_history = [
        Message('assistant', "How may I help you today?"),
        # Message("human", "Help me in some thing...")
    ]

    # User uploads :
    # This will be filled by one API call, to implement later.
    st.session_state.user_uploads = []

    # Set flag to true:
    st.session_state.initialized = True


# All variables in session state:
user_uploads, log, chat_history, server_ip = (
    st.session_state.user_uploads,
    st.session_state.logger,
    st.session_state.chat_history,
    st.session_state.server_ip
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

            @contextmanager
            def write_progress(msg: str):
                # Shared variable across multiple steps
                nonlocal progress_status

                # Start with ⏳️ to show progress:
                curr = progress_status + f"- ⏳ {msg}\n"
                container.container(border=True).markdown(curr)

                try:
                    # Do the actual step (indent of 'with')
                    yield

                    # yield is over means, step is done > Update with ✅
                    progress_status += f"\n- ✅ {msg}\n"
                    curr = progress_status

                except Exception as e:
                    progress_status += f"\n- ❌ {msg}: {e}\n"
                    raise e

                finally:
                    container.container(border=True).markdown(curr)

            try:
                for i, file in enumerate(uploaded_files):
                    progress_status += f"\n📂 Processing file {i+1} of {len(uploaded_files)}...\n"

                    # Save file:
                    with write_progress("Uploading file..."):
                        file_name = "asda"
                        time.sleep(st.secrets.llm.per_step_delay)

                    # Embed the file:
                    with write_progress("Embedding content..."):
                        time.sleep(st.secrets.llm.per_step_delay)

                    # Add file in session_state:
                    with write_progress("Finalizing the process..."):
                        st.session_state.user_uploads.append(file_name)
                        log.info(f"File `{file_name}` processed successfully.")
                        time.sleep(st.secrets.llm.end_delay)

                return True

            except Exception as e:
                log.error(f"Error processing files: {e}")
                return False


# ------------------------------------------------------------------------------
# Sidebar:
# ------------------------------------------------------------------------------

st.sidebar.toggle(label="Dummy Response Mode",
                  help="Toggle to use dummy responses instead of actual LLM responses.", value=False, key="dummy_mode")

st.sidebar.subheader("📂 Files")

selected_file = st.sidebar.selectbox(
    label="Choose File to ***Preview***",
    index=0,
    options=st.session_state.user_uploads,
)

# Tried to show pdf persistently, but it re-renders on each run and page hangs in streaming response:
if not st.session_state.user_uploads:
    st.sidebar.info("No files uploaded yet.", icon="ℹ️")
else:
    button = st.sidebar.button("Show Preview")
    if selected_file and button:
        file = os.path.join(uploads_path, selected_file)
        status, content = get_pdf_iframe(file)

        if status:
            st.sidebar.markdown(content, unsafe_allow_html=True)
        else:
            st.sidebar.error("Error loading file. Please try again.", icon="🚫")

with st.sidebar:
    st.write(st.session_state)

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
        filenames=[file.name for file in user_message.files] if user_message.files else None
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
            resp_holder = st.empty()

            # If dummy mode is enabled, use dummy response:
            if st.session_state.get("dummy_mode", False):
                response = requests.post(
                    "http://127.0.0.1:8000/rag",
                    json={
                        "query": new_message.content,
                        "session_id": st.session_state.session_id,
                        "dummy": True
                    },
                    stream=True
                )

                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        decoded = chunk.decode("utf-8")
                        decoded = json.loads(decoded)

                        if decoded["type"] == "content":
                            full += decoded["data"]
                        # elif decoded["type"] == "metadata":
                        #     full += f"```json\n{json.dumps(decoded['data'], indent=2)}\n```\n\n\n"
                        # elif decoded["type"] == "context":
                        #     documents.append(decoded['data'])
                        # else:
                        #     st.error(decoded['data'])
                        #     continue

                        resp_holder.container(border=True).markdown(full + "█")

            else:
                st.snow()
                st.stop()

                # resp = get_response_stream(new_message.content, dummy=False)
                # resp = get_conversational_rag_response(new_message.content, session_id="154")
                # for chunk in resp:
                #     # st.write(chunk)
                #     full += chunk
                #     trail_char = "█"  # "█", "▌", "|", "•"
                #     resp_holder.container(border=True).markdown(full + trail_char)

    st.session_state.chat_history.append(Message("assistant", full))
    st.rerun()
