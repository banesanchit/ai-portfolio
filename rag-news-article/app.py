import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsBot — Ask Your Articles",
    page_icon="📰",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
  }

  .main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f97316, #facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }

  .sub-title {
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
  }

  /* ── Answer box — white text ── */
  .answer-box {
    background: #1a1a1a;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1rem;
    font-size: 1rem;
    line-height: 1.7;
    color: #ffffff !important;
  }

  .source-box {
    background: #111;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #f97316;
  }

  .step-badge {
    background: #f97316;
    color: #000;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    margin-right: 8px;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background-color: #111 !important;
    border-right: 1px solid #222;
  }

  /* ── Sidebar input — white text on dark bg ── */
  section[data-testid="stSidebar"] input {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #555 !important;
    border-radius: 8px !important;
    caret-color: #f97316 !important;
  }

  section[data-testid="stSidebar"] input::placeholder {
    color: #555 !important;
  }

  section[data-testid="stSidebar"] label {
    color: #f0f0f0 !important;
  }

  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] li {
    color: #cccccc !important;
  }

  /* ── Main input ── */
  .stTextInput > div > div > input {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
  }

  .stTextInput > div > div > input::placeholder {
    color: #555 !important;
  }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(90deg, #f97316, #facc15) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 2rem !important;
    width: 100%;
  }

</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📰 NewsBot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Paste article URLs → Ask any question → Get AI answers with sources</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### <span class='step-badge'>STEP 1</span> Add Article URLs", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    url1 = st.text_input("🔗 URL 1", placeholder="https://example.com/article1")
    url2 = st.text_input("🔗 URL 2", placeholder="https://example.com/article2")
    url3 = st.text_input("🔗 URL 3", placeholder="https://example.com/article3")
    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button("⚡ Process Articles")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Paste 1–3 news URLs")
    st.markdown("2. Click Process")
    st.markdown("3. Ask any question below")
    st.markdown("4. Get answers + sources!")

# ── Session state ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "urls_processed" not in st.session_state:
    st.session_state.urls_processed = []

# ── Process Articles ──────────────────────────────────────────────────────────
if process_btn:
    urls = [u for u in [url1, url2, url3] if u.strip() != ""]
    if not urls:
        st.warning("⚠️ Please enter at least one URL in the sidebar.")
    else:
        with st.spinner("🔄 Loading and processing articles..."):
            progress = st.progress(0, text="📥 Loading articles...")
            try:
                loader = WebBaseLoader(urls)
                loader.requests_kwargs = {"verify": False}
                data = loader.load()

                if not data:
                    st.error("❌ Could not load articles. Please check URLs.")
                    st.stop()

                progress.progress(30, text=f"✅ Loaded {len(data)} article(s)!")
            except Exception as e:
                st.error(f"❌ Error loading URLs: {e}")
                st.stop()

            progress.progress(50, text="✂️ Splitting text into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=1000,
                chunk_overlap=200,
            )
            docs = splitter.split_documents(data)

            if not docs:
                st.error("❌ No content found in articles. Try different URLs.")
                st.stop()

            progress.progress(70, text="🧠 Creating embeddings (may take 30–60 sec)...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(docs, embeddings)

            progress.progress(100, text="✅ Done!")
            st.session_state.vectorstore = vectorstore
            st.session_state.urls_processed = urls
            time.sleep(0.5)
            progress.empty()

        st.success(f"✅ {len(urls)} article(s) processed! {len(docs)} chunks created. Now ask below 👇")

# ── Show loaded URLs ──────────────────────────────────────────────────────────
if st.session_state.urls_processed:
    with st.expander("📄 Currently loaded articles"):
        for i, u in enumerate(st.session_state.urls_processed, 1):
            st.markdown(f"**{i}.** {u}")

# ── Question Section ──────────────────────────────────────────────────────────
st.markdown("### <span class='step-badge'>STEP 2</span> Ask a Question", unsafe_allow_html=True)

question = st.text_input(
    label="Your question",
    placeholder="e.g. What is ChatGPT? Who founded OpenAI?",
    label_visibility="collapsed",
)

ask_btn = st.button("🔍 Get Answer")

if ask_btn:
    if not question.strip():
        st.warning("⚠️ Please type a question first.")
    elif st.session_state.vectorstore is None:
        st.warning("⚠️ Please process your articles first using the sidebar.")
    else:
        with st.spinner("🤖 Thinking..."):

            llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama-3.3-70b-versatile",
                temperature=0.3,
            )

            prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context from articles to answer the question.
Give a detailed and accurate answer based on the context.
If you cannot find the answer in the context, say "I couldn't find this in the articles."

Context:
{context}

Question: {question}

Answer:""")

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            try:
                answer = chain.invoke(question)
                st.markdown("#### 💡 Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                st.markdown("#### 🔗 Sources Used")
                for u in st.session_state.urls_processed:
                    st.markdown(f'<div class="source-box">🔗 {u}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<center style='color:#444;font-size:0.8rem'>Built with LangChain · FAISS · Groq · Streamlit</center>",
    unsafe_allow_html=True,
)