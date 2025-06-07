import json
import requests
import streamlit as st

st.set_page_config(
    page_title="Streamlit LLM Demo",
    page_icon="✨",
)
st.subheader("Streaming response Test App")

st.code("Hello, introduce yourself in xml and then tell me a joke in json.", language="markdown")

query = st.text_input(
    "Ask something", placeholder="Ask something to LLM. You can try copy-pasting the code sample above.")

c1, c2, c3, c4 = st.columns([2, 2, 2, 4], vertical_alignment='center')

c1.text("Dummy Mode:")
dummy_mode = c2.toggle(label="Dummy Mode", label_visibility='collapsed', value=False)

c3.text("Session ID:")
session_id = c4.text_input(label="Session ID", label_visibility='collapsed', value="Bbs1412")


b1, b2, _ = st.columns([2, 2, 6], vertical_alignment='center')

if b1.button("Ask Chat", icon="✨"):
    with st.spinner("⏳️ Streaming ..."):
        response = requests.post(
            "http://127.0.0.1:8000/simple/stream",
            json={"query": query, "session_id": session_id, "dummy": dummy_mode},
            stream=True
        )

        full_reply = ""
        placeholder = st.empty()

        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                decoded = chunk.decode("utf-8")
                decoded = json.loads(decoded)

                if decoded["type"] == "metadata":
                    full_reply += f"```json\n{json.dumps(decoded['data'], indent=2)}\n```\n\n\n"

                elif decoded["type"] == "content":
                    full_reply += decoded["data"]

                else:
                    st.error(decoded['data'])
                    continue

                placeholder.container(border=True).markdown(full_reply + "█")


if b2.button("Ask RAG", icon="📚"):
    with st.spinner("⏳️ Streaming ..."):
        response = requests.post(
            "http://127.0.0.1:8000/rag",
            json={"query": query, "session_id": session_id, "dummy": dummy_mode},
            stream=True
        )

        full_reply = ""
        documents = []
        document_holder = st.empty()
        reply_holder = st.empty()

        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                decoded = chunk.decode("utf-8")
                decoded = json.loads(decoded)

                if decoded["type"] == "metadata":
                    full_reply += f"```json\n{json.dumps(decoded['data'], indent=2)}\n```\n\n\n"

                elif decoded["type"] == "context":
                    documents.append(decoded['data'])

                elif decoded["type"] == "content":
                    full_reply += decoded["data"]

                else:
                    st.error(decoded['data'])
                    continue

                if documents:
                    docs = document_holder.expander("Retrieved Documents", expanded=True)
                    tabs = docs.tabs(tabs=[f"Document {i+1}" for i in range(len(documents))])
                    for i, doc in enumerate(documents):
                        with tabs[i]:
                            st.json(doc['metadata'])
                            st.markdown(doc['page_content'])

                reply_holder.container(border=True).markdown(full_reply + "█")
