import json
import requests
import streamlit as st


st.code("Hello, introduce yourself in xml and then tell me a joke in json.", language="markdown")
query = st.text_input("Ask something", placeholder="Ask something to LLM. You can try copy-pasting the code sample above.")
session_id = "Bbs1412"

dummy_mode = st.toggle("Dummy Mode", value=False)

if st.button("Ask"):
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


