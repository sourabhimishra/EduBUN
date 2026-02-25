import streamlit as st
import os
import tempfile
import time
import random
import datetime
from gtts import gTTS
import speech_recognition as sr

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ---------------- 1. PAGE CONFIG (MUST BE FIRST & ONLY ONCE) ----------------
st.set_page_config(page_title="EduBUN | AI Learning Platform", layout="wide", page_icon="🎓")

# ---------------- API ----------------
GROQ_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_KEY

# ---------------- 2. INTEGRATED CSS ----------------
def local_css():
    st.markdown("""
    <style>
        /* Modern Dark Background */
        .stApp {
            background: radial-gradient(circle at top right, #0a192f, #020617);
            color: #e2e8f0;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        /* Centering and Alignment for Professional Feel */
        .block-container {
            padding-top: 2rem;
            max-width: 900px !important;
        }

        /* Sleek Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #020617 !important;
            border-right: 1px solid rgba(0, 212, 255, 0.1);
        }

        /* Professional Glassmorphism */
        .main-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 3rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            margin-bottom: 2rem;
            text-align: center;
        }

        /* Typography */
        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.05em !important;
            background: linear-gradient(135deg, #ffffff 0%, #00d4ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem !important;
        }

        h2, h3 {
            color: #00d4ff !important;
            font-weight: 600 !important;
        }

        /* Clean Professional Buttons */
        div.stButton > button {
            background-color: transparent !important;
            color: #00d4ff !important;
            border: 1px solid #00d4ff !important;
            border-radius: 8px !important;
            padding: 0.6rem 2rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            text-transform: none !important;
        }

        div.stButton > button:hover {
            background-color: rgba(0, 212, 255, 0.1) !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
            border: 1px solid #ffffff !important;
            color: #ffffff !important;
        }

        /* UI Elements Styling */
        .stTextInput input, .stSelectbox div {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 8px !important;
        }

        /* Hide elements for cleaner UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

local_css()

# ---------------- 3. STATE MANAGEMENT ----------------
if "page" not in st.session_state:
    st.session_state.page = 1
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "active_mode" not in st.session_state:
    st.session_state.active_mode = None

# ---------------- 4. UTILITY FUNCTIONS ----------------
def daily_thought():
    thoughts = [
        "Education is the most powerful weapon which you can use to change the world.",
        "Learning never exhausts the mind.",
        "The beautiful thing about learning is nobody can take it away from you.",
        "Success is the sum of small efforts repeated daily.",
        "Knowledge is power."
    ]
    today = datetime.date.today().toordinal()
    return thoughts[today % len(thoughts)]

@st.cache_resource
def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)

@st.cache_resource
def process_pdf(_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file.getbuffer())
        path = tmp.name
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)

def speak_text(text):
    try:
        tts = gTTS(text=text[:500], lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            st.audio(tmp.name, autoplay=True)
    except:
        pass

def generate_quiz(vectorstore, llm, num, topic):
    docs = vectorstore.similarity_search(topic, k=5)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Generate {num} MCQ questions on topic '{topic}'. Format: Q1. Question? A. Option B. Option C. Option D. Option ANSWER: A\nContext: {context}"
    response = llm.invoke(prompt)
    raw = response.content.split("Q")
    return ["Q" + q.strip() for q in raw if "ANSWER:" in q]

def generate_important_questions(vectorstore, llm, topic, marks):
    docs = vectorstore.similarity_search(topic, k=5)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Generate 10 important {marks} questions from topic '{topic}'.\nContext:\n{context}"
    return llm.invoke(prompt).content

def generate_meme(vectorstore, llm, topic):
    docs = vectorstore.similarity_search(topic, k=3)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Create a funny educational meme on '{topic}'.\nContext:\n{context}"
    return llm.invoke(prompt).content

# --- NEW: INTERVIEW PREP FUNCTIONS ---
def generate_interview_question(vectorstore, llm, topic, difficulty):
    docs = vectorstore.similarity_search(topic, k=3)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"You are a technical interviewer. Based on the following context, ask one challenging {difficulty} level interview question about '{topic}'.\nContext: {context}"
    return llm.invoke(prompt).content

def evaluate_interview_answer(llm, question, answer):
    prompt = f"Question: {question}\nStudent Answer: {answer}\nCritique the answer. Provide a score out of 10 and suggest the 'Ideal Answer'."
    return llm.invoke(prompt).content

# ---------------- 5. NAVIGATION & PAGE LOGIC ----------------

llm = get_llm()

# --- LANDING PAGE ---
if st.session_state.page == 1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.title("🎓 EduBUN")
    st.write("Context-Aware Doubt Resolution for the Modern Student")
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In"):
            st.session_state.page = 2
            st.rerun()
    with col2:
        if st.button("Sign Up", type="primary"):
            st.session_state.page = 3
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- SIGN IN ---
elif st.session_state.page == 2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Welcome Back")
    email = st.text_input("Email", placeholder="student@university.edu")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.session_state.logged_in = True
        st.session_state.page = 4
        st.rerun()
    if st.button("← Back"):
        st.session_state.page = 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- SIGN UP ---
elif st.session_state.page == 3:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Join EduTech")
    st.text_input("Enter Email")
    st.text_input("Choose Password", type="password")
    if st.button("Create Account"):
        st.success("Account Created Successfully!")
        st.session_state.page = 4
        st.rerun()
    if st.button("← Back"):
        st.session_state.page = 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- ONBOARDING ---
elif st.session_state.page == 4:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("Tailor Your Experience")
    selection = st.selectbox("Select Course/Exam:", ["JEE Mains", "NEET", "CS50 - Computer Science", "UPSC Prep", "Class 12 Boards"])
    st.info(f"You are now enrolled in: *{selection}*")
    if st.button("Start Resolving Doubts", type="primary"):
        st.session_state.page = 5
        st.rerun()
    if st.button("Logout"):
        st.session_state.page = 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN RAG APPLICATION (PAGE 5) ---
elif st.session_state.page == 5:
    
    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown("## 🎛 EduBUN Control Panel")
        uploaded_file = st.file_uploader("📂 Upload Syllabus (PDF)", type="pdf")
        if uploaded_file and "vectorstore" not in st.session_state:
            with st.spinner("Processing syllabus..."):
                st.session_state.vectorstore = process_pdf(uploaded_file)

        st.markdown("---")
        # ADDED "Interview Prep" to the radio list
        mode = st.radio("Select Mode", ["Ask Doubt", "MCQ Battle", "Meme Generator", "Important Questions", "Interview Prep"])
        if st.button("Activate Mode"):
            st.session_state.active_mode = mode

        st.markdown("---")
        st.markdown("## 🗣 Pronunciation")
        p_word = st.text_input("Enter word/topic")
        if st.button("Play Sound"):
            speak_text(p_word)

        if st.button("Logout"):
            st.session_state.clear()
            st.session_state.page = 1
            st.rerun()

    # ---------------- MAIN PANEL ----------------
    if "vectorstore" not in st.session_state:
        st.markdown("## 🎓 Welcome to EduBUN")
        st.info(daily_thought())
        st.warning("Upload your syllabus PDF in the sidebar to begin learning 🚀")
    else:
        vectorstore = st.session_state.vectorstore

        # Score Display for MCQ
        if "quiz" in st.session_state:
            st.markdown(f'<div class="score-box">🔥 Score: {st.session_state.quiz.get("score",0)}</div>', unsafe_allow_html=True)

        # MCQ BATTLE
        if st.session_state.active_mode == "MCQ Battle":
            st.markdown("## ⚔ MCQ Battle")
            topic = st.text_input("Enter Topic")
            q_num = st.number_input("Number of Questions", 1, 20, 5)
            if st.button("Start Quiz"):
                st.session_state.quiz = {"questions": generate_quiz(vectorstore, llm, q_num, topic), "current_index": 0, "score": 0, "completed": False}
            
            if "quiz" in st.session_state and not st.session_state.quiz["completed"]:
                q = st.session_state.quiz
                idx = q["current_index"]
                if idx < len(q["questions"]):
                    parts = q["questions"][idx].split("ANSWER:")
                    st.markdown(parts[0])
                    choice = st.radio("Choose:", ["A", "B", "C", "D"], key=f"q_{idx}")
                    if st.button("Lock Answer"):
                        if choice.upper() == parts[1].strip().upper(): q["score"] += 10
                        else: q["score"] -= 5
                        q["current_index"] += 1
                        st.rerun()
                else:
                    st.success(f"Final Score: {q['score']}")
                    q["completed"] = True

        # IMPORTANT QUESTIONS
        elif st.session_state.active_mode == "Important Questions":
            st.markdown("## 📚 Important Questions")
            topic = st.text_input("Enter Topic")
            marks = st.selectbox("Select Marks", ["2 Marks", "5 Marks", "10 Marks"])
            if st.button("Generate"):
                st.write(generate_important_questions(vectorstore, llm, topic, marks))

        # MEME GENERATOR
        elif st.session_state.active_mode == "Meme Generator":
            st.markdown("## 😂 Meme Generator")
            topic = st.text_input("Enter Topic")
            if st.button("Generate Meme"):
                st.info(generate_meme(vectorstore, llm, topic))

        # --- ADDED: INTERVIEW PREP MODE ---
        elif st.session_state.active_mode == "Interview Prep":
            st.markdown("## 🎙️ AI Interview Prep")
            col1, col2 = st.columns(2)
            with col1:
                topic_int = st.text_input("Interview Topic (e.g., Python)")
            with col2:
                level = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Expert"])

            if st.button("Generate Question"):
                st.session_state.current_interview_q = generate_interview_question(vectorstore, llm, topic_int, level)

            if "current_interview_q" in st.session_state:
                st.info(f"**Question:** {st.session_state.current_interview_q}")
                answer_int = st.text_area("Your Answer:", height=150)
                if st.button("Submit Answer"):
                    with st.spinner("Analyzing..."):
                        feedback = evaluate_interview_answer(llm, st.session_state.current_interview_q, answer_int)
                        st.markdown("### 📝 Evaluation")
                        st.success(feedback)

        # ---------------- ASK DOUBT (Default/Last Mode) ----------------
        elif st.session_state.active_mode == "Ask Doubt" or st.session_state.active_mode is None:
            st.markdown("## 💬 Ask Your Doubt")
            if "messages" not in st.session_state: st.session_state.messages = []
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

            audio_file = st.audio_input("Record your question")
            voice_prompt = None
            
            if audio_file:
                with st.spinner("Transcribing..."):
                    r = sr.Recognizer()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    try:
                        with sr.AudioFile(tmp_path) as source:
                            r.adjust_for_ambient_noise(source)
                            data = r.record(source)
                            voice_prompt = r.recognize_google(data)
                    except:
                        st.error("Audio error.")
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)

            text_prompt = st.chat_input("Or type your question...")
            final_prompt = voice_prompt if voice_prompt else text_prompt

            if final_prompt:
                if not st.session_state.messages or st.session_state.messages[-1]["content"] != final_prompt:
                    st.session_state.messages.append({"role": "user", "content": final_prompt})
                    with st.chat_message("assistant"):
                        docs = vectorstore.similarity_search(final_prompt, k=3)
                        context = "\n".join([d.page_content for d in docs])
                        template = ChatPromptTemplate.from_template("Context:\n{context}\nQuestion:\n{question}")
                        response = (template | llm).invoke({"context": context, "question": final_prompt})
                        st.markdown(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                        speak_text(response.content)
                    st.rerun()