# 🎥 YouTube Video Assistant

An AI-powered YouTube assistant built with **Python, LangChain, Google Gemini, FAISS, and Gradio**.

The application allows users to summarize YouTube videos and ask questions about their content using Retrieval-Augmented Generation (RAG).

## ✨ Features

- **YouTube Transcript Extraction:** Automatically retrieves English transcripts using YouTubeTranscriptApi.
- **AI Video Summarization:** Generates concise and informative video summaries using Google Gemini.
- **Question Answering:** Ask questions about video content and receive context-aware answers.
- **Semantic Search:** Uses Gemini embeddings to understand the meaning of questions and transcript sections.
- **Vector Search:** Uses FAISS to efficiently retrieve the most relevant transcript chunks.
- **Interactive Interface:** Provides a simple web interface using Gradio.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Gradio | Web interface |
| YouTubeTranscriptApi | YouTube transcript extraction |
| LangChain | LLM orchestration and processing pipelines |
| Google Gemini | Text generation and summarization |
| Gemini Embeddings | Text vectorization |
| FAISS | Vector storage and similarity search |

## 🏗️ Architecture

The application provides two main workflows.

### 1. Video Summarization

```text
YouTube URL
    |
    v
Extract Video ID
    |
    v
Fetch English Transcript
    |
    v
Process Transcript
    |
    v
LangChain Prompt Template
    |
    v
Google Gemini
    |
    v
Generated Summary
```

### 2. Question Answering (RAG)

```text
YouTube URL
    |
    v
Fetch Transcript
    |
    v
Process Transcript
    |
    v
Split Transcript into Chunks
    |
    v
Generate Embeddings
    |
    v
Store Vectors in FAISS
    |
    v
User Question
    |
    v
Semantic Similarity Search
    |
    v
Retrieve Top 5 Relevant Chunks
    |
    v
Build Context
    |
    v
Google Gemini
    |
    v
Context-Aware Answer
```

## 📋 Requirements

- Python 3.10 or later
- Google Gemini API key
- Internet connection

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -U gradio youtube-transcript-api langchain-core \
    langchain-google-genai langchain-text-splitters \
    langchain-community faiss-cpu
```

### 4. Configure environment variables

Get your API key from [Google AI Studio](https://aistudio.google.com/).

Set the environment variable:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Optional model configuration:

```bash
export GEMINI_CHAT_MODEL="gemini-3.8-flash"
export GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
```

The application uses these default models if the optional variables are not provided.

### 5. Run the application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://localhost:7860
```

## 💻 Usage

### Summarize a YouTube Video

1. Open the application.
2. Paste a valid YouTube video URL.
3. Click **Summarize Video**.
4. Wait for Gemini to process the transcript.
5. Read the generated summary.

### Ask Questions About a Video

1. Paste a YouTube video URL.
2. Enter a question about the video.
3. Click **Ask a Question**.
4. The application retrieves relevant transcript chunks using FAISS.
5. Gemini generates an answer based on the retrieved context.

## ⚙️ Configuration

The application includes the following configurable parameters:

| Parameter | Default Value | Description |
|---|---:|---|
| `CHUNK_SIZE` | 1000 | Maximum text chunk size |
| `CHUNK_OVERLAP` | 150 | Overlapping characters between chunks |
| `RETRIEVAL_K` | 5 | Number of relevant chunks retrieved |
| `CHAT_MODEL` | gemini-3.8-flash | Gemini generation model |
| `EMBEDDING_MODEL` | gemini-embedding-001 | Gemini embedding model |

## 🧠 Key Concepts

This project demonstrates several important concepts in AI engineering:

- **Large Language Models (LLMs):** Generate natural-language summaries and answers.
- **Prompt Engineering:** Guides model behavior through structured instructions.
- **Text Chunking:** Divides large transcripts into smaller manageable sections.
- **Embeddings:** Represent text as numerical vectors.
- **Semantic Search:** Retrieves information based on meaning rather than exact keyword matching.
- **Vector Search:** Uses FAISS to find similar vectors efficiently.
- **Retrieval-Augmented Generation (RAG):** Combines information retrieval with LLM generation.
- **LangChain Expression Language (LCEL):** Connects prompts, models, and output parsers into processing pipelines.

## ⚠️ Limitations

- Currently supports English transcripts only.
- Videos must have accessible transcripts.
- Summarization sends the entire processed transcript in one request, which can be problematic for very long videos.
- The FAISS index is created again for every question.
- The application does not persist vector indexes between sessions.
- Answer quality depends on transcript quality and retrieval relevance.
- Google Gemini API usage may be subject to quotas and rate limits.

## 🔮 Future Improvements

- [ ] Support multiple transcript languages.
- [ ] Implement map-reduce summarization for long videos.
- [ ] Cache transcripts and FAISS indexes.
- [ ] Add persistent vector storage.
- [ ] Improve retrieval using similarity thresholds and reranking.
- [ ] Support conversational memory for follow-up questions.
- [ ] Add video metadata extraction.
- [ ] Improve error handling and logging.

## 📚 Documentation

- [LangChain](https://python.langchain.com/docs/introduction/)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gradio](https://www.gradio.app/docs)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)

## 👨‍💻 Author

Developed as a practical project to explore **LLM integration, LangChain, embeddings, vector search, and Retrieval-Augmented Generation (RAG)**.

---

**Built with Python ❤️ and Google Gemini.**

