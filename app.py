import json
import time
import requests
import streamlit as st
from contextlib import contextmanager
from typing import Optional, List, Literal

# import server.logger as logger


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


# Get user_id:
if "session_id" not in st.session_state:

    try:
        if requests.get(f"{st.secrets.server.ip_address}/").status_code != 200:
            raise Exception
    except:
        st.error(
            "Server is not reachable. Please check your connection or server status.", icon="🚫"
        )
        st.stop()

    # with st.container(border=True, height=500):
    with st.container(border=True):
        tabs = st.tabs(tabs=['Register', 'Login'])

        with tabs[1]:
            st.header("🔐 :orange[Existing User Login]")
            ip_user_id = st.text_input(
                "User ID:", placeholder="Enter your user ID here...",
                icon="👤", key="login_user_id"
            )
            ip_user_pw = st.text_input(
                "Password:", placeholder="Enter your password here...",
                type="password", icon="🔑", key="login_user_pw"
            )

            if st.button("Login", type="primary"):
                ip_user_id = "_".join(ip_user_id.strip().lower().split(" "))
                ip_user_pw = ip_user_pw.strip()

                if not ip_user_id or not ip_user_pw:
                    st.error("Please fill all the fields.", icon="🚫")
                else:
                    # Send to server for login:
                    try:
                        resp = requests.post(
                            f"{st.secrets.server.ip_address}/login",
                            json={"login_id": ip_user_id, "password": ip_user_pw}
                        )

                        if resp.status_code == 200:
                            session_id = resp.json().get("user_id")
                            st.session_state.session_id = session_id
                            name_of_user = resp.json().get("name", session_id)
                            st.session_state.name_of_user = name_of_user

                            st.success("Login successful!", icon="✅")
                            st.rerun()
                        else:
                            st.error(resp.json().get("error", "Login failed."), icon="🚫")
                            st.stop()

                    except requests.RequestException as e:
                        st.error(f"Error connecting to server: {e}", icon="🚫")
                        st.stop()

        with tabs[0]:
            st.header("🔐 :orange[New User Registration]")
            ip_user_id = st.text_input(
                "User ID:", placeholder="Enter your user ID here...",
                icon="👤", key="register_user_id"
            )
            ip_user_name = st.text_input(
                "Your Name:", placeholder="Enter your name here...",
                icon="🔤", key="register_user_name"
            )
            st.caption("Use characters only from `lower case letters`, `numbers`, `-`, `_`")
            ip_user_pw = st.text_input(
                "Password:", placeholder="Enter your password here...",
                type="password", icon="🔑", key="register_user_pw"
            )

            if st.button("Register", type="primary"):
                ip_user_id = "_".join(ip_user_id.strip().lower().split(" "))
                ip_user_pw = ip_user_pw.strip()

                if not ip_user_name or not ip_user_id or not ip_user_pw:
                    st.error("Please fill all the fields.", icon="🚫")
                else:
                    # Send to server for registration:
                    try:
                        resp = requests.post(
                            f"{st.secrets.server.ip_address}/register",
                            json={
                                "name": ip_user_name, "user_id": ip_user_id, "password": ip_user_pw
                            }
                        )

                        if resp.status_code == 201:
                            # st.session_state.session_id = ip_user_id
                            st.success("Registration successful! You can now login.", icon="✅")
                            st.stop()
                        else:
                            st.error(resp.json().get("error", "Registration failed."), icon="🚫")
                            st.stop()

                    except requests.RequestException as e:
                        st.error(f"Error connecting to server: {e}", icon="🚫")
                        st.stop()

    st.stop()


if "initialized" not in st.session_state:
    # Initialize Logger:
    # st.session_state.logger = logger.get_logger(name="Streamlit")
    # log = st.session_state.logger
    # log.info("Streamlit initialized.")

    # Initialize the session with server::
    st.session_state.server_ip = st.secrets.server.ip_address
    try:
        resp = requests.post(
            f"{st.session_state.server_ip}/chat_history",
            data={"user_id": st.session_state.session_id}
        )
        if resp.status_code == 200:
            # Initialize messages:
            st.session_state.chat_history = [Message('assistant', "👋, How may I help you today?")]

            # Load old chat history (if):
            chat_hist = resp.json().get("chat_history", [])
            for msg in chat_hist:
                st.session_state.chat_history.append(Message(msg['role'], msg['content']))
        else:
            # log.error(f"Failed to initialize chat history: {resp.json().get('error', 'Unknown error')}")
            st.error(
                "Failed to initialize chat history. Please try again later.",
                icon="🚫"
            )
            st.stop()

    except requests.RequestException as e:
        # log.error(f"Error initializing server session: {e}")
        st.error(
            "Failed to connect to the server. Please check your connection or server status.",
            icon="🚫"
        )
        st.stop()

    # # Initialize messages:
    # st.session_state.chat_history = [
    #     Message('assistant', "👋, How may I help you today?"),
    #     # Message("human", "Help me in some thing...")
    # ]

    # User's Existing Uploads:
    st.session_state.user_uploads = requests.get(
        f"{st.session_state.server_ip}/uploads",
        params={"user_id": st.session_state.session_id}
    ).json().get("files", [])

    # Last resp retrieved docs:
    st.session_state.last_retrieved_docs = []

    # Set flag to true:
    st.session_state.initialized = True


# All variables in session state:
user_id = st.session_state.session_id
chat_history = st.session_state.chat_history
server_ip = st.session_state.server_ip
# log = st.session_state.logger


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


def upload_file(uploaded_file) -> tuple[bool, str]:
    """Upload the st attachment/uploaded file to the server and save it.
    Args:
        uploaded_file: The file object uploaded by the user.
    Returns:
        tuple: A tuple containing:
            - bool: True if the file was uploaded successfully, False otherwise.
            - str: The server file name or error message.
    """

    try:
        # POST to FastAPI
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {"user_id": user_id}
        response = requests.post(f"{server_ip}/upload", files=files, data=data)

        if response.status_code == 200:
            message = response.json().get("message", "")
            # log.info(f"File `{message}` uploaded successfully for user `{user_id}`.")
            return True, message
        else:
            message = response.json().get("error", "Unknown error")
            # log.error(
            # f"Failed to upload file `{uploaded_file.name}`: {message} for user `{user_id}`.")
            return False, message

    except Exception as e:
        # log.error(f"Error uploading file `{uploaded_file.name}`: {e} for user `{user_id}`.")
        return False, str(e)


def embed_file(file_name: str) -> tuple[bool, str]:
    """Embed the content of the file into the RAG system.
    Args:
        file_name: The name of the file to embed.
    Returns:
        tuple: A tuple containing:
            - bool: True if the file was embedded successfully, False otherwise.
            - str: Success message or error message.
    """
    try:
        response = requests.post(
            f"{server_ip}/embed",
            json={
                "user_id": user_id,
                "file_name": file_name
            }
        )

        if response.status_code == 200:
            # log.info(f"File `{file_name}` embedded successfully for user `{user_id}`.")
            return True, response.json().get("message", "File embedded successfully.")
        else:
            error_message = response.json().get("error", "Unknown error")
            # log.error(f"Failed to embed file `{file_name}`: {error_message} for user `{user_id}`.")
            return False, error_message

    except Exception as e:
        # log.error(f"Error embedding file `{file_name}`: {e} for user `{user_id}`.")
        return False, str(e)


def handle_uploaded_files(uploaded_files) -> bool:
    """Handle the uploaded files by uploading them to the server and embedding their content."""
    progress_status = ""

    with st.chat_message(name='assistant', avatar='./assets/settings_3.png'):
        with st.spinner("Processing files..."):
            container = st.empty()

            # Found out later that all this thing can be done with st.status() as status:
            # But, it does not allow that much customization.
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
                    # log.info(f"Processing file: {file.name}")

                    # Upload file:
                    with write_progress("Uploading file..."):
                        status, message = upload_file(file)
                        if not status:
                            raise RuntimeError(f"Upload failed for file: {file.name}")
                        server_file_name = message
                        time.sleep(st.secrets.llm.per_step_delay)

                    # Embed the file:
                    with write_progress("Embedding content..."):
                        status, message = embed_file(server_file_name)
                        if not status:
                            raise RuntimeError(f"Embedding failed for file: {file.name}")
                        time.sleep(st.secrets.llm.per_step_delay)

                    # Any last steps like finalizing or cleanup:
                    with write_progress("Finalizing the process..."):
                        # Update data with latest user_upload
                        st.session_state.user_uploads = requests.get(
                            f"{st.session_state.server_ip}/uploads",
                            params={"user_id": user_id}
                        ).json().get("files", [])

                        # log.info(f"File `{file.name}` processed successfully.")
                        time.sleep(st.secrets.llm.end_delay)

                return True

            except Exception as e:
                st.exception(exception=e)
                st.stop()
                return False


@st.cache_data(ttl=60 * 10, show_spinner=False)
def get_iframe(file_name: str, num_pages: int = 5) -> tuple[bool, str]:
    """Get the iframe HTML for the PDF file."""
    try:
        response = requests.post(
            f"{st.session_state.server_ip}/iframe",
            json={
                "user_id": user_id,
                "file_name": file_name,
                "num_pages": num_pages
            },
        )
        if response.status_code == 200:
            return True, response.json().get("iframe", "")
        else:
            return False, response.json().get("error", "Unknown error")
    except requests.RequestException as e:
        # log.error(f"Error getting iframe for {file_name}: {e}")
        return False, str(e)


# ------------------------------------------------------------------------------
# Sidebar:
# ------------------------------------------------------------------------------

# User Profile:
with st.sidebar.container(border=True):
    c1, c2 = st.columns([1, 8])
    # c1.image("./assets/user.png", use_container_width=True)
    c1.write("👤")
    # c2.write(" ".join([word.capitalize() for word in user_id.split("_")]))
    c2.write(st.session_state.get("name_of_user", "Error Occurred!"))
# st.sidebar.divider()


# Files Preview:
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
        status, content = get_iframe(selected_file)
        if status:
            st.sidebar.markdown(content, unsafe_allow_html=True)
        else:
            st.sidebar.error(f"Error: **{content}**", icon="🚫")
st.sidebar.divider()

# Dummy Mode Toggle:
st.sidebar.toggle(label="Dummy Response Mode", value=False, key="dummy_mode",
                  help="Toggle to use dummy responses instead of actual LLM responses.")

# Clear My Data:
if st.sidebar.button("Clear My Uploads", type="secondary", icon="🗑️"):
    try:
        resp = requests.post(
            f"{st.session_state.server_ip}/clear_my_files",
            data={"user_id": user_id}
        )
        if resp.status_code == 200:
            st.success(resp.json().get("message", "Uploads cleared successfully!"), icon="✅")
        else:
            st.error(resp.json().get("error", "Failed to clear Uploads."), icon="🚫")
    except requests.RequestException as e:
        st.error(f"Error clearing Uploads: {e}", icon="🚫")

# Clear my Chat History:
if st.sidebar.button("Clear My Chat History", type="secondary", icon="💬"):
    resp = requests.post(
        f"{server_ip}/clear_chat_history",
        data={"user_id": user_id}
    )

    if resp.status_code == 200:
        st.session_state.chat_history = [
            Message('assistant', "👋, How may I help you today?")
        ]
        st.session_state.last_retrieved_docs = []
        st.success("Chat history cleared successfully!", icon="✅")
    else:
        st.error(resp.json().get("error", "Failed to clear chat history."), icon="🚫")

# with st.sidebar:
#     st.write(st.session_state)

# ------------------------------------------------------------------------------
# Page content:
# ------------------------------------------------------------------------------

a, b = st.columns([0.65, 9.35], vertical_alignment='bottom', gap='small')
a.image("./assets/gemma.jpg", use_container_width=True)
b.header(":green[RAG] with :blue[Gemma-3]", divider='rainbow')


for ind, message in enumerate(st.session_state.chat_history):
    if ind < len(st.session_state.chat_history) - 1:                # all messages except last
        if message.type == 'human':
            write_as_human(message.content, message.filenames)

        elif message.type == 'assistant':
            answer = message.content
            if "<think>" in answer:
                answer = answer[answer.find("</think>") + len("</think>"):]
            write_as_ai(answer)

    else:                                                           # Last message
        if message.type == 'human':                                 # if human, write normally
            write_as_human(message.content)

        elif message.type == 'assistant':                           # if assistant
            # Get the answer, thoughts and docs from the message:
            full = message.content
            thoughts = full[
                full.find("<think>")+8:full.find("</think>")
            ] if "<think>" in full else None
            answer = full[full.find("</think>") + len("</think>"):] if thoughts else full
            documents = st.session_state.last_retrieved_docs if st.session_state.last_retrieved_docs else None

            with st.chat_message(name='assistant', avatar='assistant'):
                with st.container(border=True):
                    # # Thinking:
                    # if thoughts:
                    #     cont_thoughts = st.expander("💭 Thoughts", expanded=True).markdown(thoughts)
                    # # Answer:
                    # st.markdown(answer)
                    # # Documents:
                    # if documents:
                    #     tabs = st.expander("🗃️ Sources", expanded=False).tabs(
                    #         tabs=[f"Document {i+1}" for i in range(len(documents))]
                    #     )
                    #     for i, doc in enumerate(documents):
                    #         with tabs[i]:
                    #             st.subheader(":blue[Content:]")
                    #             st.markdown(doc['page_content'])
                    #             st.divider()
                    #             st.subheader(":blue[Source Details:]")
                    #             st.json(doc['metadata'], expanded=False)

                    # Thinking:
                    if thoughts:
                        # cont_thoughts = c1.expander("💭 Thoughts", expanded=True).markdown(thoughts)
                        cont_thoughts = st.popover(
                            "💭 Thoughts", use_container_width=False).markdown(thoughts)
                    # Answer:
                    st.markdown(answer)
                    # Documents:
                    if documents:
                        tabs = st.expander("🗃️ Sources", expanded=False).tabs(
                            tabs=[f"Document {i+1}" for i in range(len(documents))]
                        )
                        for i, doc in enumerate(documents):
                            with tabs[i]:
                                st.subheader(":blue[Content:]")
                                st.markdown(doc['page_content'])
                                st.divider()
                                st.subheader(":blue[Source Details:]")
                                st.json(doc['metadata'], expanded=False)


if user_message := st.chat_input(
    placeholder="Enter any queries here... You can also attach [pdf, txt, md] files.",
    max_chars=1000,
    accept_file='multiple',
    file_type=['pdf', 'txt', 'md'],
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
    # Clear last documents:
    st.session_state.last_retrieved_docs = []

    # Handle the files if any:
    if user_message.files:
        if handle_uploaded_files(user_message.files):
            st.toast("Files processed successfully!", icon="✅")
        else:
            st.error("Error processing files. Please try again.", icon="🚫")

    # Get response and write it:
    with st.chat_message(name='assistant', avatar='assistant'):
        with st.spinner("Generating response..."):
            full = ""

            # If dummy mode is enabled, use dummy response:
            if st.session_state.get("dummy_mode", False):
                resp_holder = st.empty()
                response = requests.post(
                    f"{server_ip}/rag",
                    json={
                        "query": new_message.content,
                        "session_id": user_id,
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

                        resp_holder.markdown(full + "█")

            else:                                           # real RAG response from server
                response = requests.post(
                    f"{server_ip}/rag",
                    json={
                        "query": new_message.content,
                        "session_id": user_id,
                        "dummy": False
                    },
                    stream=True
                )

                documents = []
                resp_holder = st.container(border=True)
                document_holder = resp_holder.empty()
                reply_holder = resp_holder.empty()

                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        decoded = chunk.decode("utf-8")
                        decoded = json.loads(decoded)

                        if decoded["type"] == "metadata":
                            # Skip metadata for now
                            continue
                            # full += f"```json\n{json.dumps(decoded['data'], indent=2)}\n```\n\n\n"

                        elif decoded["type"] == "context":
                            documents.append(decoded['data'])

                        elif decoded["type"] == "content":
                            full += decoded["data"]

                        else:
                            st.error(decoded['data'])
                            continue

                        if documents:
                            docs = document_holder.expander("🗃️ Sources", expanded=True)
                            tabs = docs.tabs(
                                tabs=[f"Document {i+1}" for i in range(len(documents))])
                            for i, doc in enumerate(documents):
                                with tabs[i]:
                                    st.subheader(":blue[Content:]")
                                    st.markdown(doc['page_content'])
                                    st.divider()
                                    st.subheader(":blue[Source Details:]")
                                    st.json(doc['metadata'], expanded=False)

                        reply_holder.container(border=True).markdown(full + "█")

                st.session_state.last_retrieved_docs = documents
            st.session_state.chat_history.append(Message("assistant", full))
    st.rerun()
