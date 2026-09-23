import streamlit as st
from agent import agent_executor
import os
import subprocess

if not os.path.exists("chroma_db"):
    with st.spinner("Setting up knowledge base for the first time..."):
        subprocess.run(["python", "ingest.py"])

st.set_page_config(page_title="Job Market RAG Agent", page_icon="💼")

st.title("💼 Job Market Research Assistant")
st.caption("Ask about saved job postings or live market info — the agent decides which source to use.")

# Keep chat history across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "tool" in msg:
            st.caption(f"🔧 Tool used: {msg['tool']}")

# Input box
query = st.chat_input("Ask a question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": query})
            answer = response["output"]

            # Extract which tool(s) were used
            tools_used = []
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    tools_used.append(step[0].tool)
            tool_label = ", ".join(set(tools_used)) if tools_used else "none"

            st.write(answer)
            st.caption(f"🔧 Tool used: {tool_label}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool": tool_label
    })
    