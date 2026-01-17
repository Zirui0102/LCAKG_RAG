import streamlit as st
import uuid
from lca_qa import build_qna
from langchain_openai import ChatOpenAI

# Page config
st.set_page_config(
    page_title="LCA QA",
    page_icon="🧠",
    layout="wide"
)

# Sidebar: configuration
st.sidebar.header("🔑 Configuration")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password"
)
neo4j_url = st.sidebar.text_input(
    "Neo4j URL", value="bolt://localhost:7687"
)
neo4j_user = st.sidebar.text_input(
    "Neo4j Username", value="neo4j"
)
neo4j_password = st.sidebar.text_input(
    "Neo4j Password", type="password"
)

# Interpretation toggle
interpret_answer = st.sidebar.checkbox(
    "🧠 Interpret answer (ChatGPT)", value=True
)

init_clicked = st.sidebar.button("🚀 Initialize")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ask" not in st.session_state:
    st.session_state.ask = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "interpreter" not in st.session_state:
    st.session_state.interpreter = None

# Header
st.markdown(
    """
    <h1 style="
        color: #7FBAD6;
        margin-bottom: 0;
    ">
        LCAKG Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: -6px;
    ">
        Ask questions about the LCA-oriented Knowledge Graph
    </p>
    """,
    unsafe_allow_html=True
)

# Initialization
if init_clicked:
    if not (openai_api_key and neo4j_url and neo4j_user and neo4j_password):
        st.sidebar.error("Please fill in all fields.")
    else:
        with st.spinner("Initializing engine…"):
            try:
                st.session_state.ask = build_qna(
                    openai_api_key=openai_api_key,
                    neo4j_url=neo4j_url,
                    neo4j_username=neo4j_user,
                    neo4j_password=neo4j_password,
                )

                # ChatGPT interpreter (separate, lightweight)
                st.session_state.interpreter = ChatOpenAI(
                    model="gpt-4o",
                    temperature=1,
                    openai_api_key=openai_api_key,
                )

                st.sidebar.success("✅ Initialized! You can start asking questions.")
            except Exception as e:
                st.sidebar.error(f"Initialization failed:\n{e}")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input & execution
if st.session_state.ask is None:
    st.info("Enter credentials in the sidebar and click **Initialize**.")
else:
    if prompt := st.chat_input("Ask a question about your graph..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Querying knowledge graph..."):
                try:
                    raw_answer = st.session_state.ask(
                        prompt,
                        thread_id=st.session_state.thread_id
                    )
                except Exception as e:
                    raw_answer = f"⚠️ Engine error:\n```\n{e}\n```"

            if interpret_answer and st.session_state.interpreter:
                with st.spinner("🧠 Interpreting result..."):
                    interpretation_prompt = f"""
You are an expert in Life Cycle Assessment (LCA).

The following is a factual answer retrieved from a Neo4j-based LCA knowledge graph.

TASK:
- Explain and interpret the answer in clear, concise scientific language.
- Use ONLY the information present in the answer.
- Do NOT introduce new data, assumptions, or conclusions.
- If the answer is a list or table, summarize key patterns.
- If numerical values are present, explain what they represent.

ANSWER:
{raw_answer}
"""
                    interpreted = st.session_state.interpreter.invoke(
                        interpretation_prompt
                    ).content.strip()

                st.markdown(interpreted)
                st.session_state.messages.append(
                    {"role": "assistant", "content": interpreted}
                )

                # Optional: show raw answer
                with st.expander("📄 Show raw database answer"):
                    st.markdown(raw_answer)

            else:
                # No interpretation
                st.markdown(raw_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": raw_answer}
                )

# Sidebar utilities
with st.sidebar.expander("🧪 Session Info"):
    st.write("Thread ID:", st.session_state.thread_id)
    st.write("Messages:", len(st.session_state.messages))

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.experimental_rerun()
