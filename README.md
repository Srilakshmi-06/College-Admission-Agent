# 🎓 AI College Admission Agent
### RAG-Based Multi-Agent System for Intelligent Admission Guidance

> **An intelligent digital admission counselor** that helps students with course selection, eligibility checking, fees, scholarships, required documents, deadlines, and application procedures — powered by Retrieval-Augmented Generation (RAG) and a Multi-Agent Architecture.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Multi-Agent System](#multi-agent-system)
5. [RAG Pipeline](#rag-pipeline)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Environment Setup](#environment-setup)
10. [Ollama Setup (Local LLM)](#ollama-setup-local-llm)
11. [IBM Granite / watsonx.ai Setup (Optional)](#ibm-granite--watsonxai-setup-optional)
12. [Knowledge Base Ingestion](#knowledge-base-ingestion)
13. [Running the Application](#running-the-application)
14. [Testing](#testing)
15. [Troubleshooting](#troubleshooting)
16. [Future Scope](#future-scope)

---

## 📌 Project Overview

The **AI College Admission Agent** is a production-quality, locally-runnable AI application that acts as a personalized admission counselor. It combines:

- **RAG (Retrieval-Augmented Generation)** — Every answer is grounded in actual admission documents
- **Multi-Agent Architecture** — 5 specialized agents handle different admission topics
- **FAISS Vector Search** — Fast semantic retrieval over the knowledge base
- **Anti-Hallucination Design** — The system never invents fees, deadlines, or eligibility criteria
- **Source Transparency** — Every answer shows which document it came from
- **Student Profiling** — Personalizes recommendations based on academic background

---

## 🎯 Problem Statement

Students navigating college admissions face several challenges:

- Information is scattered across multiple documents and websites
- Eligibility criteria are complex and vary by course and category
- Scholarship information is hard to find and verify
- Deadlines are easy to miss
- Getting personalized guidance requires visiting the college in person

---

## 💡 Solution Architecture

```
Student Query
      ↓
Streamlit UI (Professional Web Interface)
      ↓
Query Router (Intent Classification)
      ↓
Specialized Agent (selected by category)
      ↓
RAG Retriever
      ↓
FAISS Vector Database (local)
      ↓
Relevant Admission Document Chunks
      ↓
IBM Granite / Ollama LLM
      ↓
Grounded Response + Source Citations
      ↓
Student Receives Personalized Guidance
```

For multi-domain queries, the **Agent Coordinator** runs multiple agents and synthesizes a unified response.

---

## 🤖 Multi-Agent System

| Agent | Category | Responsibility |
|---|---|---|
| **Admission Knowledge Agent** | FAQ, Deadlines, General | General Q&A, admission policies, important dates |
| **Course Recommendation Agent** | COURSE | Personalized course advice based on student profile |
| **Eligibility Agent** | ELIGIBILITY | Eligibility checking against course requirements |
| **Application Guidance Agent** | APPLICATION, DOCUMENTS | Step-by-step procedures and document checklists |
| **Scholarship & Fee Agent** | FEES, SCHOLARSHIP | Fee structures and financial aid information |

The **Query Router** classifies queries using keyword pattern matching and routes them to the correct agent. Multi-category queries trigger **Agent Collaboration** where multiple agents contribute to a single synthesized response.

---

## 📚 RAG Pipeline

```
Documents (PDF / DOCX / TXT)
      ↓
Document Loader (rag/loader.py)
      ↓
Text Extraction + Cleaning
      ↓
Document Chunker (rag/chunker.py)
  → chunk_size=600 chars, overlap=100 chars
  → Sentence-aware splitting
  → Rich metadata per chunk
      ↓
Embedding Model (rag/embeddings.py)
  → sentence-transformers/all-MiniLM-L6-v2
  → 384-dimensional dense vectors
      ↓
FAISS Index (rag/vector_store.py)
  → IndexFlatIP (cosine similarity)
  → Persisted to vectorstore/
      ↓
Retriever (rag/retriever.py)
  → top-k similarity search
  → Deduplication
  → Source formatting
```

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| **LLM (Primary)** | IBM Granite (`granite-3.2-8b-instruct`) via watsonx.ai |
| **LLM (Local Fallback)** | Ollama (llama3.2 or any supported model) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | FAISS (faiss-cpu) |
| **Document Parsing** | pypdf, python-docx |
| **Web Interface** | Streamlit |
| **Development Env** | IBM Bob |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
college-admission-agent/
│
├── app.py                      # Main Streamlit application
├── config.py                   # Central configuration management
├── ingest.py                   # CLI knowledge base ingestion script
├── requirements.txt
├── .env.example                # Environment variable template
├── README.md
│
├── data/                       # Knowledge base documents (replace with official docs)
│   ├── courses.txt
│   ├── eligibility.txt
│   ├── fees.txt
│   ├── admission_policy.txt
│   ├── important_dates.txt
│   ├── scholarships.txt
│   ├── documents_required.txt
│   └── faq.txt
│
├── vectorstore/                # FAISS index (auto-generated by ingest.py)
│
├── agents/                     # Multi-agent system
│   ├── __init__.py
│   ├── router.py               # Query classification and routing
│   ├── base_agent.py           # Abstract base class for all agents
│   ├── coordinator.py          # Multi-agent orchestration
│   ├── knowledge_agent.py      # General admission knowledge
│   ├── course_agent.py         # Course recommendations
│   ├── eligibility_agent.py    # Eligibility checking
│   ├── application_agent.py    # Application guidance
│   └── scholarship_agent.py    # Fees and scholarships
│
├── rag/                        # RAG pipeline
│   ├── __init__.py
│   ├── loader.py               # Document loading (PDF/DOCX/TXT)
│   ├── chunker.py              # Text chunking with metadata
│   ├── embeddings.py           # Sentence transformer embeddings
│   ├── vector_store.py         # FAISS vector store
│   └── retriever.py            # Query retrieval and ranking
│
├── llm/                        # LLM provider abstraction
│   ├── __init__.py
│   ├── base.py                 # Abstract base interface
│   ├── local.py                # Ollama provider
│   ├── granite.py              # IBM Granite / watsonx.ai provider
│   └── factory.py              # Provider factory function
│
├── components/                 # Streamlit UI components
│   ├── __init__.py
│   ├── ui.py                   # CSS and layout components
│   ├── chat.py                 # Chat message rendering
│   ├── cards.py                # Feature cards and stats
│   └── sources.py              # Source citation display
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── document_loader.py      # High-level ingestion utilities
│   └── validators.py           # File and input validation
│
└── tests/                      # Test scripts
    └── test_pipeline.py
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip
- [Ollama](https://ollama.com) (for local LLM — recommended)
- Git (optional)

### Windows Setup

```powershell
# 1. Clone or navigate to the project directory
cd CollegeGuide_AI_College_Admission_Agent

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### macOS / Linux Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔧 Environment Setup

```powershell
# Copy the example .env file
copy .env.example .env
```

Edit `.env` with your configuration:

```env
# LLM Provider: "local" (Ollama) or "ibm" (watsonx.ai)
LLM_PROVIDER=local

# Ollama settings (used when LLM_PROVIDER=local)
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Embedding model (used for all providers)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# RAG settings
TOP_K=4
CHUNK_SIZE=600
CHUNK_OVERLAP=100
```

---

## 🦙 Ollama Setup (Local LLM)

Ollama runs LLMs locally on your machine with no internet required for inference.

### 1. Install Ollama

Download from: **https://ollama.com**

For Windows: Download and run the installer.

### 2. Start Ollama Server

```powershell
ollama serve
```

### 3. Pull the Model

```powershell
ollama pull llama3.2
```

Other supported models:
```powershell
ollama pull mistral
ollama pull llama3.1
ollama pull gemma2
```

Update `OLLAMA_MODEL` in your `.env` to match the model you pulled.

### 4. Verify Ollama

```powershell
ollama list
```

---

## 🔷 IBM Granite / watsonx.ai Setup (Optional)

To use IBM Granite as the LLM instead of Ollama:

### 1. Install the IBM watsonx.ai SDK

```powershell
pip install ibm-watsonx-ai
```

### 2. Configure Credentials

Edit your `.env` file:

```env
LLM_PROVIDER=ibm
IBM_CLOUD_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL=ibm-granite/granite-3.2-8b-instruct
```

### 3. Obtain Credentials

1. Sign in at [IBM Cloud](https://cloud.ibm.com)
2. Create a watsonx.ai service instance
3. Copy your API key from IBM Cloud IAM
4. Copy your project ID from the watsonx.ai dashboard

> **Note:** The application falls back to local Ollama if IBM credentials are missing or invalid, so local development always works without IBM Cloud.

---

## 📥 Knowledge Base Ingestion

Before running the application, build the knowledge base:

```powershell
# Ingest all documents from the data/ directory
python ingest.py

# Check current index status
python ingest.py --status

# Ingest from a different directory
python ingest.py --dir path/to/your/documents

# Add a single document
python ingest.py --file path/to/document.pdf

# Clear the index and start fresh
python ingest.py --clear
```

**Expected output:**
```
==============================================================
  AI College Admission Agent — Knowledge Base Ingestion
==============================================================

  Source directory : data/
  Documents found  : 7

  Starting ingestion pipeline...

  [████████████████████████████░░] 95%  Building vector index...
  [██████████████████████████████] 100% Knowledge base ready!

  ✅ Knowledge base built successfully!
  📄 Documents indexed : 7
  🧩 Chunks created   : 86
```

### Adding Your Own Documents

Replace the sample `.txt` files in `data/` with your institution's official documents:

```powershell
# Copy your PDF documents to data/
copy "C:\path\to\Official_Course_Catalog.pdf" data\
copy "C:\path\to\Admission_Brochure_2025.pdf" data\

# Rebuild the knowledge base
python ingest.py
```

Supported formats: **PDF**, **DOCX**, **TXT**

---

## 🚀 Running the Application

```powershell
# Make sure the virtual environment is active
venv\Scripts\activate

# Run the Streamlit application
streamlit run app.py
```

The application opens automatically in your browser at **http://localhost:8501**

### Navigation

| Page | Description |
|---|---|
| 🏠 **Home** | Landing page with feature overview and quick stats |
| 🤖 **Admission Assistant** | Main chat interface — ask admission questions |
| 📚 **Knowledge Base** | Upload documents, rebuild index, view stats |
| ℹ️ **About** | Project information, architecture, technology |

---

## 🧪 Testing

Run the test suite:

```powershell
python tests/test_pipeline.py
```

### Manual Test Queries

Open the Admission Assistant and try:

| Query | Expected Agent |
|---|---|
| "Which courses are available?" | Course Recommendation Agent |
| "Am I eligible for B.Tech Computer Science?" | Eligibility Agent |
| "What is the tuition fee?" | Scholarship & Fee Agent |
| "What scholarships are available?" | Scholarship & Fee Agent |
| "What documents do I need?" | Application Guidance Agent |
| "When is the application deadline?" | Admission Knowledge Agent |
| "How do I apply?" | Application Guidance Agent |
| "I studied Maths and CS and I like AI. Which course?" | Multi-Agent Collaboration |
| "What is the hostel fee for international students?" | Should say: not in documents |

---

## 🔧 Troubleshooting

### Problem: Ollama not running

**Symptom:** `Cannot connect to Ollama at http://localhost:11434`

**Fix:**
```powershell
# Start Ollama
ollama serve

# In a new terminal, verify it works
ollama list
```

### Problem: Model not found in Ollama

**Symptom:** `Model 'llama3.2' not found in Ollama`

**Fix:**
```powershell
ollama pull llama3.2
```

### Problem: Knowledge base is empty

**Symptom:** `Knowledge base not found` warning in the app

**Fix:**
```powershell
python ingest.py
```

### Problem: sentence-transformers takes a long time on first run

**Explanation:** The embedding model (~90 MB) is downloaded from Hugging Face on first use. This is a one-time download that takes 1–3 minutes. Subsequent runs are fast.

### Problem: FAISS installation fails on Windows

**Fix:**
```powershell
pip install faiss-cpu --prefer-binary
```

### Problem: pypdf fails to extract text from a PDF

**Explanation:** Some PDFs are image-based (scanned documents) and cannot be parsed. Convert to text-based PDF using OCR tools like Adobe Acrobat or Tesseract before ingesting.

### Problem: IBM credentials not working

**Check:**
1. `IBM_CLOUD_API_KEY` is set correctly (no quotes, no spaces)
2. `WATSONX_PROJECT_ID` is your project UUID, not your project name
3. `WATSONX_URL` matches your IBM Cloud region
4. The `ibm-watsonx-ai` package is installed: `pip install ibm-watsonx-ai`

The app will automatically fall back to Ollama if IBM credentials are invalid.

---

## 🚀 Future Scope

| Feature | Description |
|---|---|
| **Real-Time College Portal Integration** | Connect to official APIs for live admission data |
| **Voice Assistant** | Voice input and text-to-speech responses |
| **Multilingual Support** | Tamil, Hindi, and other regional languages |
| **Mobile App** | React Native / Flutter companion app |
| **Application Tracker** | Track application status across colleges |
| **Email Notifications** | Deadline reminders sent to students |
| **Admin Dashboard** | Manage documents and view analytics |

---

## 🔐 Security Notes

- Never commit your `.env` file (it is in `.gitignore_template`)
- Never hardcode API keys in source files
- Uploaded files are validated for type and size before saving
- File paths are sanitized to prevent directory traversal
- Internal prompts and system instructions are not exposed to users

---

## 📄 License

This project is created for educational and demonstration purposes.
For production use, review and comply with the licenses of all dependencies.

---

## 🙏 Acknowledgements

- **IBM Bob** — Development and orchestration environment
- **IBM Granite** — LLM backbone (when configured)
- **Meta AI / Facebook AI Research** — FAISS vector search library
- **Hugging Face** — sentence-transformers and model hub
- **Streamlit** — Web application framework
- **Ollama** — Local LLM serving

---

*Built with ❤️ using IBM Bob · IBM Granite · Agentic AI · RAG · FAISS · Streamlit*
