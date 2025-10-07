from dotenv import load_dotenv
import os
from pathlib import Path
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
load_dotenv()

# Robust .env loading: try src/.env (relative to this file), then project root .env, then default search
project_root = Path(__file__).parent
loaded = False

src_env = project_root / "src" / ".env"
if src_env.exists():
    load_dotenv(dotenv_path=src_env)
    loaded = True

if not loaded:
    root_env = project_root / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=root_env)
        loaded = True

if not loaded:
    load_dotenv()

# Diagnostics
print("Using env from:", src_env if src_env.exists() else ((project_root / ".env") if (project_root / ".env").exists() else "default search"))
print("Has PINECONE_API_KEY after load:", bool(os.getenv("PINECONE_API_KEY")))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing after attempting multiple .env locations.\n"
                       f"Checked: {src_env} and {project_root / '.env'}.\n"
                       "Add the key to one of these files or export it in your shell.")
# GROQ_API_KEY is only needed for LLM usage, not for Pinecone indexing here.

# 1) Load and prepare documents
extracted_data = load_pdf_files(data='data')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# 2) Get embeddings model
embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired

if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)