import streamlit as st
import requests

query = st.text_input("Ask something")
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
                full_reply += decoded
                placeholder.container(border=True).markdown(full_reply)

