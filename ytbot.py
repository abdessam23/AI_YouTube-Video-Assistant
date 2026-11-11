"""
YouTube transcript summarizer and Q&A app
-----------------------------------------
Uses:
- YouTubeTranscriptApi to fetch English transcripts
- LangChain text splitters
- Google Gemini for summaries and answers
- Gemini embeddings + FAISS for retrieval
- Gradio for the user interface

Set GEMINI_API_KEY in your environment before running:
    export GEMINI_API_KEY="your-api-key"

Install dependencies:
    pip install -U gradio youtube-transcript-api langchain-core \
        langchain-google-genai langchain-text-splitters \
        langchain-community faiss-cpu
"""

import os
import re
from urllib.parse import parse_qs, urlparse
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5

def get_api_key() -> str:
    """Read and validate the Gemini API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Set it in your environment and restart the app."
        )
    return api_key


def initialize_gemini() -> ChatGoogleGenerativeAI:
    """Create a LangChain-compatible Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=get_api_key(),
        temperature=0.2,
    )


def setup_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Create a LangChain-compatible Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=get_api_key(),
    )


# ---------------------------------------------------------------------------
# YouTube transcript extraction
# ---------------------------------------------------------------------------

def get_video_id(url: str) -> str | None:
    """Extract an 11-character YouTube video ID from common YouTube URL forms."""
    if not url or not url.strip():
        return None

    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    video_id = None

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        query_id = parse_qs(parsed.query).get("v")
        if query_id:
            video_id = query_id[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                video_id = parts[1]

    if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return video_id

    return None


def get_transcript(url: str):
    """
    Fetch an English transcript.
    The current youtube-transcript-api interface returns a FetchedTranscript
    whose entries expose .text and .start attributes.
    """
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("That does not look like a valid YouTube video URL.")

    transcript_api = YouTubeTranscriptApi()
    return transcript_api.fetch(video_id, languages=["en"])


def process(transcript) -> str:
    """Convert transcript snippets into one plain-text string."""
    lines = []

    for item in transcript:
        text = getattr(item, "text", "")
        if text:
            # Remove line breaks that can occur inside individual captions.
            lines.append(text.replace("\n", " ").strip())

    processed_text = " ".join(line for line in lines if line)
    if not processed_text:
        raise ValueError("The English transcript is empty.")

    return processed_text


# ---------------------------------------------------------------------------
# Chunking and vector search
# ---------------------------------------------------------------------------

def chunk_transcript(
    processed_transcript: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split a transcript into overlapping text chunks."""
    if not processed_transcript.strip():
        raise ValueError("There is no transcript text to split.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(processed_transcript)


def create_faiss_index(chunks: list[str], embedding_model):
    """Embed transcript chunks and store them in a FAISS vector index."""
    if not chunks:
        raise ValueError("No transcript chunks were created.")
    return FAISS.from_texts(chunks, embedding_model)


def retrieve(query: str, faiss_index, k: int = RETRIEVAL_K):
    """Retrieve the most relevant transcript chunks for a question."""
    return faiss_index.similarity_search(query, k=k)


# ---------------------------------------------------------------------------
# LangChain prompts and chains
# --------------------------------------------------------------------------------------

def create_summary_chain():
    """Create a modern LCEL chain for transcript summarization."""
    prompt = PromptTemplate.from_template(
        """
You are an assistant that summarizes YouTube video transcripts.

Write one concise but informative summary paragraph. Capture the main ideas,
important explanations, and conclusions. Do not invent information. Do not
include timestamps.

Transcript:
{transcript}
"""
    )

    return prompt | initialize_gemini() | StrOutputParser()


def create_qa_chain():
    """Create a modern LCEL chain for retrieval-based question answering."""
    prompt = PromptTemplate.from_template(
        """
You are an assistant answering questions about a YouTube video.

Use only the video context below to answer the question. If the context does
not contain enough information, clearly say that the transcript does not
provide the answer. Do not invent facts.

Video context:
{context}

Question:
{question}

Answer:
"""
    )

    return prompt | initialize_gemini() | StrOutputParser()


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def summarize_video(video_url: str):
    """Fetch a video's transcript and return its summary."""
    try:
        transcript = get_transcript(video_url)
        processed_transcript = process(transcript)

        summary_chain = create_summary_chain()
        summary = summary_chain.invoke({"transcript": processed_transcript})

        return summary, "Summary generated successfully."
    except Exception as exc:
        return "", f"Could not summarize this video: {exc}"


def answer_question(video_url: str, user_question: str):
    """Build a transcript index, retrieve relevant chunks, and answer a question."""
    try:
        if not user_question or not user_question.strip():
            return "", "Please enter a question about the video."

        transcript = get_transcript(video_url)
        processed_transcript = process(transcript)

        chunks = chunk_transcript(processed_transcript)
        embedding_model = setup_embedding_model()
        faiss_index = create_faiss_index(chunks, embedding_model)

        relevant_documents = retrieve(user_question, faiss_index)
        context = "\n\n".join(
            document.page_content for document in relevant_documents
        )

        qa_chain = create_qa_chain()
        answer = qa_chain.invoke(
            {
                "context": context,
                "question": user_question,
            }
        )

        return answer, "Answer generated successfully."
    except Exception as exc:
        return "", f"Could not answer the question: {exc}"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="YouTube Video Assistant") as interface:
    gr.Markdown("# YouTube Video Assistant")
    gr.Markdown(
        "Paste a YouTube URL to summarize its English transcript or ask "
        "questions about the video."
    )

    video_url_input = gr.Textbox(
        label="YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    with gr.Row():
        summarize_button = gr.Button("Summarize Video", variant="primary")

    summary_output = gr.Textbox(
        label="Video Summary",
        lines=7,
        interactive=False,
    )

    question_input = gr.Textbox(
        label="Ask a Question About the Video",
        placeholder="What are the main points discussed?",
        lines=2,
    )

    question_button = gr.Button("Ask a Question")
    answer_output = gr.Textbox(
        label="Answer",
        lines=7,
        interactive=False,
    )

    status_output = gr.Textbox(
        label="Status",
        interactive=False,
    )

    summarize_button.click(
        fn=summarize_video,
        inputs=[video_url_input],
        outputs=[summary_output, status_output],
    )

    question_button.click(
        fn=answer_question,
        inputs=[video_url_input, question_input],
        outputs=[answer_output, status_output],
    )


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)

