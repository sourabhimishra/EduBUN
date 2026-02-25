# EduBUN
EduBUN is an AI-powered, syllabus-aware learning platform built using RAG architecture. It enables PDF-based doubt solving, MCQ quizzes, interview prep, voice interaction, and contextual AI responses using Groq LLM, HuggingFace embeddings, and ChromaDB for semantic search.
🎓 EduBUN – AI-Powered Context-Aware Learning Platform

EduBUN is a syllabus-aware AI learning assistant built using Retrieval-Augmented Generation (RAG). It allows students to upload their syllabus PDF and receive intelligent, context-based answers strictly derived from their study material.

The system processes uploaded PDFs by splitting them into semantic chunks and converting them into embeddings using HuggingFace models. These embeddings are stored in a Chroma vector database, enabling fast similarity search. When a user asks a question (via text or voice), the system retrieves the most relevant content and generates a precise answer using the Groq-hosted LLaMA 3.3 70B language model.

🚀 Key Features

📂 PDF-based contextual doubt resolution

💬 AI Chat Tutor (Ask Doubt mode)

⚔ MCQ Battle with scoring system

📚 Important Question Generator (2/5/10 marks)

🎙 AI Interview Preparation with evaluation

🔊 Voice input & Text-to-Speech support

🌑 Modern dark UI with glassmorphism design

⚡ Optimized performance using @st.cache_resource

🛠 Tech Stack

Streamlit (Frontend + Backend)

LangChain (LLM orchestration)

Groq LLM (LLaMA 3.3 70B)

HuggingFace Embeddings

Chroma Vector Database

SpeechRecognition

gTTS (Google Text-to-Speech)

EduBUN aims to evolve into a scalable AI mentor platform that supports adaptive learning, performance analytics, and institutional deployment.
