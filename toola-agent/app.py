import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent — Tools",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0a0f;
    color: #e0e0ff;
  }
  .main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .sub-title { font-size: 1rem; color: #666; margin-bottom: 2rem; }
  .answer-box {
    background: #12121f;
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1rem;
    font-size: 1rem;
    line-height: 1.8;
    color: #e0e0ff;
  }
  .thinking-box {
    background: #0d0d1a;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.4rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #a5b4fc;
  }
  .tool-used-box {
    background: #1a0d2e;
    border: 1px solid #7c3aed;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: #c4b5fd;
  }
  .step-box {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #a5b4fc;
    text-align: center;
  }
  .stTextInput > div > div > input {
    background: #12121f !important;
    color: #e0e0ff !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 8px !important;
  }
  .stButton > button {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 2rem !important;
    width: 100%;
  }
  section[data-testid="stSidebar"] {
    background-color: #0d0d1a !important;
    border-right: 1px solid #1a1a3a;
  }

  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] li,
  section[data-testid="stSidebar"] label {
    color: #e0e0ff !important;
  }

  section[data-testid="stSidebar"] h3 {
    color: #a855f7 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🤖 AI Tools Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask anything → AI picks the right tool → Gets you the answer</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Available Tools")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🔍 Wikipedia**")
    st.markdown("Facts about people, places, events, science")
    st.markdown("---")
    st.markdown("**🌐 DuckDuckGo Search**")
    st.markdown("Current news and recent web information")
    st.markdown("---")
    st.markdown("**🧮 Calculator**")
    st.markdown("Solves any math problem")
    st.markdown("---")
    st.markdown("### 💡 Try These Questions")
    st.markdown("- Who is Sam Altman?")
    st.markdown("- What is quantum computing?")
    st.markdown("- What is 125 * 48 + 200?")
    st.markdown("- Latest news about AI?")
    st.markdown("- Who invented the telephone?")
    st.markdown("- What is machine learning?")

# ── Define Tools ──────────────────────────────────────────────────────────────
@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual information about any topic, person, place or event."""
    try:
        wiki = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=1500
        )
        return wiki.run(query)
    except Exception as e:
        return f"Wikipedia search failed: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web for current news, recent events and latest information."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                output = ""
                for r in results:
                    output += f"Title: {r['title']}\n"
                    output += f"Summary: {r['body']}\n\n"
                return output
            return "No results found"
    except Exception as e:
        return f"Web search failed: {e}"

@tool
def calculator(expression: str) -> str:
    """Solve math calculations. Input must be a math expression like 25 * 48 + 100"""
    try:
        allowed = set('0123456789+-*/().% ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        else:
            return "Error: Only math expressions allowed!"
    except Exception as e:
        return f"Calculation error: {e}"

# ── Build Agent ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )
    tools = [wikipedia_search, web_search, calculator]

    # ✅ system prompt tells agent exactly which tools to use
    system_prompt = """You are a helpful assistant with access to exactly these 3 tools:
1. wikipedia_search - for facts about people, places, history, science
2. web_search - for current news, recent events, latest information  
3. calculator - for math calculations only

IMPORTANT RULES:
- ONLY use the tools listed above
- For news/current events use web_search
- For facts/history use wikipedia_search
- For math use calculator
- NEVER use any other tool like brave_search or google_search
- Always give a clear final answer"""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
    return agent

# ── Main Question Area ────────────────────────────────────────────────────────
st.markdown("### 💬 Ask Anything")

question = st.text_input(
    label="question",
    placeholder="e.g. Who is Sam Altman? / What is 25*48? / Latest AI news",
    label_visibility="collapsed",
)

ask_btn = st.button("🚀 Get Answer")

if ask_btn:
    if not question.strip():
        st.warning("⚠️ Please type a question first.")
    else:
        with st.spinner("🤖 Agent thinking and using tools..."):
            try:
                agent = get_agent()

                result = agent.invoke({
                    "messages": [HumanMessage(content=question)]
                })

                messages = result.get("messages", [])

                # ── Show thinking process ──
                with st.expander("🔍 Agent Thinking Process", expanded=True):
                    for msg in messages:
                        if isinstance(msg, HumanMessage):
                            st.markdown(
                                f'<div class="thinking-box">👤 <b>You:</b> {msg.content}</div>',
                                unsafe_allow_html=True
                            )
                        elif isinstance(msg, AIMessage):
                            if msg.content:
                                st.markdown(
                                    f'<div class="thinking-box">🤖 <b>Agent:</b> {msg.content}</div>',
                                    unsafe_allow_html=True
                                )
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    st.markdown(
                                        f'<div class="tool-used-box">🛠️ <b>Tool:</b> {tc["name"]} | <b>Input:</b> {tc["args"]}</div>',
                                        unsafe_allow_html=True
                                    )
                        elif isinstance(msg, ToolMessage):
                            st.markdown(
                                f'<div class="thinking-box">📦 <b>Tool Result:</b> {str(msg.content)[:300]}...</div>',
                                unsafe_allow_html=True
                            )

                # ── Final Answer ──
                final_answer = messages[-1].content if messages else "No answer found."
                st.markdown("#### 💡 Final Answer")
                st.markdown(
                    f'<div class="answer-box">{final_answer}</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ── How it works ──────────────────────────────────────────────────────────────
st.divider()
st.markdown("### ⚙️ How This Agent Works")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        '<div class="step-box">STEP 1<br><br>You ask question<br>↓<br>Agent reads it</div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        '<div class="step-box">STEP 2<br><br>Agent decides<br>which tool to use<br>↓<br>Wikipedia / Search / Calculator</div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        '<div class="step-box">STEP 3<br><br>Tool gets data<br>↓<br>Agent forms answer<br>↓<br>You get result!</div>',
        unsafe_allow_html=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<center style='color:#444;font-size:0.8rem'>Built with LangGraph · Groq · Wikipedia · DuckDuckGo · Streamlit</center>",
    unsafe_allow_html=True,
)