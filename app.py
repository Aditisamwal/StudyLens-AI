import streamlit as st
import fitz
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Page Configuration
st.set_page_config(
    page_title="StudyLens AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Aesthetic CSS (Soft Pastel Lavender & Plum Theme)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #f3e8ff 50%, #fce7f3 100%);
        color: #2d1b4e;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border-right: 1px solid #e9d5ff;
    }
    .main-header {
        background: linear-gradient(90deg, #6b21a8, #be185d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .ai-card {
        background: linear-gradient(135deg, #ffffff 0%, #faf5ff 100%);
        padding: 22px;
        border-radius: 16px;
        border: 2px solid #a855f7;
        box-shadow: 0 6px 20px rgba(147, 51, 234, 0.12);
        margin-bottom: 20px;
        color: #1f2937;
        line-height: 1.7;
    }
    .answer-card {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #9333ea;
        box-shadow: 0 4px 12px rgba(147, 51, 234, 0.06);
        margin-bottom: 14px;
        color: #1f2937;
        line-height: 1.6;
    }
    .match-badge {
        display: inline-block;
        background-color: #f3e8ff;
        color: #7e22ce;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 10px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7e22ce, #db2777);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def extract_pdf_data(file_bytes):
    document = fitz.open(stream=file_bytes, filetype="pdf")
    num_pages = len(document)
    text = ""
    for page in document:
        text += page.get_text() + "\n"
    return text, num_pages

def split_text(text):
    chunks = re.split(r'\n+', text)
    return [c.strip() for c in chunks if len(c.strip()) > 40]

def get_top_keywords(text, top_n=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in ENGLISH_STOP_WORDS]
    common = Counter(filtered).most_common(top_n)
    return pd.DataFrame(common, columns=["Keyword", "Frequency"]).set_index("Keyword")

def generate_summary(chunks, num_points=5):
    if len(chunks) <= num_points:
        return chunks
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)
    scores = tfidf_matrix.sum(axis=1).A1
    top_idx = scores.argsort()[-num_points:][::-1]
    return [chunks[i] for i in sorted(top_idx)]

def ask_groq(api_key, context, question):
    client = Groq(api_key=api_key)
    prompt = f"""
    You are StudyLens AI, a helpful study assistant.
    Answer the student's question clearly and concisely using ONLY the provided PDF context below.
    If the answer is not in the context, say "Ye topic uploaded PDF mein nahi mila."

    PDF Context:
    {context}

    Student Question:
    {question}
    """
    available_models = [
        m.id for m in client.models.list().data
        if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "playai"])
    ]
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=available_models[0],
    )
    return chat_completion.choices[0].message.content

# Sidebar Section
with st.sidebar:
    st.markdown("## 🤖 AI Settings")
    api_key = st.text_input("Groq API Key (Free)", type="password", placeholder="Paste your gsk_... key here")
    if api_key:
        st.success("API Key Connected! ✅")
    st.caption("Key nahi hai? [console.groq.com](https://console.groq.com) se free lein.")

    st.markdown("---")
    st.markdown("## 📂 Document Hub")
    uploaded_file = st.file_uploader("Upload your Study PDF", type=["pdf"])

    if uploaded_file:
        text, num_pages = extract_pdf_data(uploaded_file.getvalue())
        chunks = split_text(text)
        word_count = len(text.split())
        read_time = max(1, round(word_count / 200))

        st.success("PDF Ready!")
        st.markdown("### 📊 Document Stats")
        col1, col2 = st.columns(2)
        col1.metric("Pages", num_pages)
        col2.metric("Words", word_count)
        st.metric("Est. Reading Time", f"{read_time} min")
        st.metric("Text Sections", len(chunks))
    else:
        st.info("Upload a PDF here to unlock all study tools.")

# Main Header
st.markdown('<p class="main-header">✨ StudyLens AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your aesthetic, smart PDF study companion — Powered by RAG & Llama 3 AI.</p>', unsafe_allow_html=True)

# Main Content Area
if uploaded_file:
    tab1, tab2, tab3 = st.tabs(["🤖 Ask AI", "📝 Quick Summary", "📈 Keyword Insights"])

    # TAB 1: AI Q&A (RAG)
    with tab1:
        st.markdown("#### Ask anything from your uploaded notes")
        question = st.text_input("Type your question:", placeholder="e.g., Explain the main concept in simple words...")

        if st.button("✨ Ask StudyLens AI"):
            if question.strip() and chunks:
                vectorizer = TfidfVectorizer(stop_words="english")
                vectors = vectorizer.fit_transform(chunks + [question])
                scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

                top_indices = scores.argsort()[-4:][::-1]
                relevant_chunks = [chunks[i] for i in top_indices if scores[i] > 0]
                if not relevant_chunks and chunks:
                    relevant_chunks = chunks[:8]
                if relevant_chunks:
                    combined_context = "\n\n".join(relevant_chunks)
                    saved_notes = f"Question: {question}\n\n"

                    if api_key.strip():
                        with st.spinner("✨ AI is reading your notes and writing an answer..."):
                            try:
                                ai_answer = ask_groq(api_key.strip(), combined_context, question)
                                st.markdown(f"""
                                <div class="ai-card">
                                    <span class="match-badge">✨ AI Generated Answer</span>
                                    <div>{ai_answer}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                saved_notes += f"AI Answer:\n{ai_answer}\n\n"
                            except Exception as e:
                                st.error(f"AI Error: API key check karein. ({e})")
                    else:
                        st.info("💡 Sidebar mein Groq API key paste karein taaki AI direct answer likh sake.")

                    with st.expander("📖 View Source Paragraphs from PDF", expanded=(not bool(api_key.strip()))):
                        for rank, index in enumerate(top_indices, 1):
                            if scores[index] > 0:
                                match_pct = round(scores[index] * 100, 1)
                                ans_text = chunks[index]
                                saved_notes += f"Source {rank} ({match_pct}% Match):\n{ans_text}\n\n"
                                st.markdown(f"""
                                <div class="answer-card">
                                    <span class="match-badge">🎯 {match_pct}% Match</span>
                                    <div>{ans_text}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    st.download_button(
                        label="📥 Download Q&A Notes (.txt)",
                        data=saved_notes,
                        file_name="studylens_ai_notes.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No matching text found in the PDF. Try using keywords from your document.")
            else:
                st.warning("Please enter a question first.")

    # TAB 2: Auto Summary
    with tab2:
        st.markdown("#### Key Takeaways from Your PDF")
        num_points = st.slider("Select number of key points:", 3, 8, 5)
        if st.button("⚡ Generate Summary"):
            summary_points = generate_summary(chunks, num_points)
            for i, point in enumerate(summary_points, 1):
                st.markdown(f"""
                <div class="answer-card">
                    <span class="match-badge">Point {i}</span>
                    <div>{point}</div>
                </div>
                """, unsafe_allow_html=True)

    # TAB 3: Keyword Insights
    with tab3:
        st.markdown("#### Most Important Topics & Keywords")
        if text.strip():
            kw_df = get_top_keywords(text, top_n=10)
            st.bar_chart(kw_df, color="#9333ea")

            with st.expander("📄 Preview Extracted PDF Text"):
                st.write(text[:2000] + ("..." if len(text) > 2000 else ""))
else:
    st.markdown("""
    <div class="answer-card" style="text-align: center; padding: 40px;">
        <h3>👈 Start by uploading a PDF from the left sidebar</h3>
        <p style="color: #6b7280;">Upload lecture slides, textbook chapters, or notes to chat with AI, generate instant summaries, and visualize key topics.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("✨ StudyLens AI | Built with Python, Streamlit, Scikit-Learn & Groq Llama 3")