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

st.set_page_config(page_title="Smart Agriculture Assistant", page_icon="🌱", initial_sidebar_state="expanded")


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

CSS = """
<style>
.hero{background:linear-gradient(135deg,#1B5E20 0%,#2E7D32 55%,#66A94D 100%);color:#fff;
      padding:28px 32px;border-radius:18px;margin-bottom:18px;box-shadow:0 6px 18px rgba(27,94,32,.25)}
.hero-title{font-size:2rem;font-weight:800;line-height:1.2}
.hero-sub{opacity:.92;margin-top:6px;font-size:1.02rem}
.chips{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.chips span{background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.35);
            padding:4px 12px;border-radius:999px;font-size:.9rem}
[data-testid="stChatMessage"]{background:#fff;border:1px solid #dbe7d3;border-radius:16px;
            padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,.05)}
.stButton>button{border-radius:12px;border:1px solid #2E7D32;color:#1B5E20;background:#fff;text-align:left}
.stButton>button:hover{background:#E8F5E9;border-color:#1B5E20;color:#1B5E20}
footer{visibility:hidden}
</style>
"""

HERO = """
<div class="hero">
  <div class="hero-title">🌱 Smart Agriculture Assistant</div>
  <div class="hero-sub">Practical, source-backed advice for smallholder farmers in Rwanda</div>
  <div class="chips"><span>🌽 Maize</span><span>🫘 Beans</span><span>🍠 Cassava</span><span>🥔 Potato</span><span>🍌 Banana</span></div>
</div>
"""

AVATARS = {"user": "🧑‍🌾", "assistant": "🌱"}
SOURCES_MARK = "\n\n**Sources**\n"


def render_reply(content):
    if SOURCES_MARK in content:
        answer, sources = content.split(SOURCES_MARK, 1)
        st.markdown(answer)
        with st.expander(f"📚 Sources ({len(sources.splitlines())})"):
            st.markdown(sources)
    else:
        st.markdown(content)


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

crop_counts = {}
for c in chunks:
    crop = str(c.get("crop") or "other")
    crop_counts[crop] = crop_counts.get(crop, 0) + 1
n_docs = len({c.get("file") or c.get("source") for c in chunks})

with st.sidebar:
    st.markdown("### 🌾 About")
    st.write("Ask farming questions and get answers drawn only from trusted guides, "
             "with sources you can check.")
    st.markdown("### 📚 Knowledge base")
    st.write(f"{n_docs} documents · {len(chunks)} passages")
    for crop, n in sorted(crop_counts.items()):
        st.write(f"- {crop.title()}: {n} passages")
    st.markdown("### 💡 Tips")
    st.write("- Name the crop\n- Describe what you see on the plant\n- Ask one question at a time")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.rerun()

question = st.chat_input("Ask a question about your crops...")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None
if question:
    st.session_state.messages.append({"role": "user", "content": question})

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        if m["role"] == "assistant":
            render_reply(m["content"])
        else:
            st.markdown(m["content"])

if question:
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Searching the farming guides..."):
            reply = answer_question(question)
        render_reply(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
elif not st.session_state.messages:
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown("**Muraho! 👋** I'm your farming assistant. Ask me about growing, pests and "
                    "diseases for maize, beans, cassava, potatoes and bananas. "
                    "Try one of these:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"example_{i}"):
            st.session_state.pending = example
            st.rerun()

st.caption("Answers come from documents from several countries. Always confirm pesticide names "
           "and doses with your local agricultural extension officer or the Rwanda Agriculture Board (RAB).")
