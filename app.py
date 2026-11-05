import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq

APP_DIR = Path(__file__).parent
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
TOP_K = 4

SYSTEM_PROMPT = """You are a smart agriculture assistant for smallholder farmers in Rwanda.
Answer the question using ONLY the numbered context passages provided.
Rules:
- If the context does not contain the answer, reply: "I don't have enough information in my documents to answer that."
- Give practical, clear advice in simple English. Use short steps or bullet points when helpful.
- Cite the passages you used, like [1] or [2].
- Never invent pesticide names, doses, dates or numbers that are not in the context.
- The documents come from several countries. If a passage names an office, agency or phone number from another country, do not tell the farmer to contact it. Say "your local agricultural extension officer or the Rwanda Agriculture Board (RAB)" instead."""

EXAMPLES = [
    "How can I control fall armyworm on my maize?",
    "What are the signs of late blight on potatoes?",
    "How do I stop banana wilt from spreading?",
    "How do I choose good cassava cuttings?",
]

st.set_page_config(page_title="Smart Agriculture Assistant", page_icon="🌱")


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_resources():
    index = faiss.read_index(str(APP_DIR / "agri.faiss"))
    with open(APP_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    return index, chunks, embedder


def retrieve(question, k=TOP_K):
    vec = np.array(list(embedder.embed([QUERY_PREFIX + question])), dtype="float32")
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    scores, ids = index.search(vec, k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i != -1]


def build_context(results):
    return "\n\n".join(
        f"[{i + 1}] (Source: {r['source']}, page {r['page']})\n{r['text']}"
        for i, r in enumerate(results)
    )


def build_messages(question, context):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def generate(question, context):
    client = Groq(api_key=get_secret("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=0.2,
        messages=build_messages(question, context),
    )
    return resp.choices[0].message.content.strip()


def answer_question(question):
    results = retrieve(question, TOP_K)
    sources = "\n".join(
        f"- [{i + 1}] {r['source']}, page {r['page']}" for i, r in enumerate(results)
    )
    try:
        answer = generate(question, build_context(results))
    except Exception as e:
        answer = ("Sorry, the answer service is busy or unavailable right now. "
                  f"Please try again in a minute. ({type(e).__name__})")
    return f"{answer}\n\n**Sources**\n{sources}"


if not get_secret("GROQ_API_KEY"):
    st.error("The GROQ_API_KEY secret is missing. Add it in the app's Settings > Secrets.")
    st.stop()

index, chunks, embedder = load_resources()

st.title("🌱 Smart Agriculture Assistant")
st.markdown(
    "Ask about maize, beans, cassava, Irish potatoes and bananas. "
    "Answers come only from trusted farming guides, with sources."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

question = st.chat_input("Ask a question about your crops...")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None
if question:
    st.session_state.messages.append({"role": "user", "content": question})

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question:
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            reply = answer_question(question)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
elif not st.session_state.messages:
    st.markdown("**Examples**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"example_{i}"):
            st.session_state.pending = example
            st.rerun()

if st.session_state.messages:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.caption("Answers come from documents from several countries. Always confirm pesticide names "
           "and doses with your local agricultural extension officer or the Rwanda Agriculture Board (RAB).")
