import streamlit as st
import requests

API = "http://127.0.0.1:8000"

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI SaaS RAG",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# SESSION
# =========================================================

if "chat" not in st.session_state:
    st.session_state.chat = []

if "token" not in st.session_state:
    st.session_state.token = None

if "username" not in st.session_state:
    st.session_state.username = None

# =========================================================
# TITLE
# =========================================================

st.title("🧠 AI SaaS RAG Platform")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🔐 Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        res = requests.post(
            f"{API}/login",
            json={
                "username": username,
                "password": password
            }
        )

        data = res.json()

        if res.status_code == 200:

            st.session_state.token = data["token"]

            st.session_state.username = data["user"]

            st.success("Login successful")

        else:

            st.error(data["detail"])

    st.divider()

    if st.session_state.username:

        st.success(
            f"User: {st.session_state.username}"
        )

        if st.button("Clear Chat"):

            st.session_state.chat = []

            st.rerun()

# =========================================================
# LAYOUT
# =========================================================

col1, col2 = st.columns([2.5, 1])

# =========================================================
# CHAT
# =========================================================

with col1:

    st.subheader("💬 Chat")

    for msg in st.session_state.chat:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "Ask something..."
    )

    if prompt:

        if not st.session_state.username:

            st.warning("Please login first")

            st.stop()

        st.session_state.chat.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                res = requests.post(
                    f"{API}/chat",
                    json={
                        "user":
                        st.session_state.username,

                        "message":
                        prompt
                    }
                )

                data = res.json()

                answer = data.get(
                    "response",
                    "No response"
                )

                st.markdown(answer)

                st.session_state.chat.append({
                    "role": "assistant",
                    "content": answer
                })

# =========================================================
# PDF UPLOAD
# =========================================================

with col2:

    st.subheader("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if not st.session_state.token:

            st.warning("Login required")

            st.stop()

        if st.button("Upload PDF"):

            headers = {
                "Authorization":
                f"Bearer {st.session_state.token}"
            }

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "application/pdf"
                )
            }

            res = requests.post(
                f"{API}/upload-pdf",
                headers=headers,
                files=files
            )

            data = res.json()

            if res.status_code == 200:

                st.success("PDF uploaded")

                st.json(data)

            else:

                st.error(data["detail"])