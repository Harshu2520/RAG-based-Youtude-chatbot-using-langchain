"""
Streamlit app: RAG over a fixed YouTube video's transcript.

Local run:
  1. pip install -r requirements.txt
  2. Create .streamlit/secrets.toml with:
       OPENROUTER_API_KEY = "your_key_here"
  3. streamlit run app.py

Deploy on Streamlit Community Cloud:
  1. Push this repo to GitHub (app.py, requirements.txt, .gitignore — NOT secrets.toml)
  2. On share.streamlit.io, create a new app pointing at this repo/app.py
  3. In the app's "Settings -> Secrets", paste:
       OPENROUTER_API_KEY = "your_key_here"

To point this at a different video, just change VIDEO_ID below.
"""

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Config — change this to point the app at a different video
# ---------------------------------------------------------------------------

VIDEO_ID = "Gfr50f6ZBvo"  # only the ID, not the full URL

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openrouter/free"
RETRIEVER_K = 4

PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
    input_variables=["context", "question"],
)


# ---------------------------------------------------------------------------
# Cached resources (avoid recomputation on every Streamlit rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=["en"])
    return " ".join(snippet.text for snippet in fetched)


@st.cache_resource(show_spinner=False)
def build_vector_store(video_id: str):
    transcript = fetch_transcript(video_id)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.create_documents([transcript])
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def format_docs(retrieved_docs) -> str:
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = st.secrets["OPENROUTER_API_KEY"]
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
    )


def build_chain(llm, retriever):
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    parser = StrOutputParser()
    return parallel_chain | PROMPT | llm | parser


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="YouTube RAG", page_icon="🎥", layout="wide")
st.title("🎥 Chat with this video")

video_col, chat_col = st.columns([1, 1.3])

with video_col:
    st.video(f"https://www.youtube.com/watch?v={VIDEO_ID}")

with chat_col:
    try:
        with st.spinner("Loading transcript and building index..."):
            vector_store = build_vector_store(VIDEO_ID)
            llm = get_llm()
    except TranscriptsDisabled:
        st.error("Captions are disabled for this video.")
        st.stop()
    except NoTranscriptFound:
        st.error("No transcript found in the requested language(s).")
        st.stop()
    except VideoUnavailable:
        st.error("This video is unavailable.")
        st.stop()
    except KeyError:
        st.error("Missing OPENROUTER_API_KEY in Streamlit secrets.")
        st.stop()

    question = st.text_input("Ask a question about the video")

    if st.button("Ask", type="primary", disabled=not question):
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": RETRIEVER_K}
        )
        chain = build_chain(llm, retriever)
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
        st.markdown("**Answer:**")
        st.write(answer)
