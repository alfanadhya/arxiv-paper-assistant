import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever_hybrid_rrf import MetadataIntent, build_retriever

load_dotenv()  # reads GOOGLE_API_KEY, QDRANT_URL, etc. from .env

if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("Set GOOGLE_API_KEY in the .env file first.")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # local Qdrant server
COLLECTION = "sebun_bge768_hybrid_full"  # built with bge-base (768 dims); must match the retriever's model
K_RETRIEVE = 20  # candidates fetched from Qdrant
K_FINAL = 5      # chunks passed to Gemini

# Fixed prompt for now — replace later
PROMPT = "Generative adverserial networks for computer security"

# max_retries: automatically retry when Gemini is busy (503) or rate-limited (429)
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),  # check AI Studio for current model names
    max_retries=5,
)

# Optional backup model, used only if the main one keeps failing
fallback_model = os.getenv("GEMINI_FALLBACK_MODEL")
if fallback_model:
    llm = llm.with_fallbacks([ChatGoogleGenerativeAI(model=fallback_model, max_retries=2)])

# 1. Hybrid search (BGE dense + BM25 sparse, fused with RRF on the Qdrant server)
print("Loading models and connecting to Qdrant...", flush=True)
retriever = build_retriever(qdrant_url=QDRANT_URL, collection_name=COLLECTION)
try:
    print("Searching Qdrant...", flush=True)
    hits = retriever.retrieve(PROMPT, K_RETRIEVE, MetadataIntent())[:K_FINAL]
finally:
    retriever.close()
print(f"Found {len(hits)} chunks.", flush=True)

if not hits:
    raise SystemExit("No matching documents found in Qdrant.")

# 2. Build context from the retrieved chunks
context = "\n\n".join(
    f"[{i}] {hit.title}\n{hit.url}\n{hit.text}" for i, hit in enumerate(hits, start=1)
)

# 3. Ask Gemini to answer using only that context
messages = [
    ("system",
     "retrieve three best mathcing papars."
     "Cite sources as [1], [2], etc. If the context doesn't contain the answer, say so.\n\n"
     f"Context:\n{context}"),
    ("human", PROMPT),
]
print("Asking Gemini...", flush=True)
try:
    response = llm.invoke(messages)
except Exception as exc:
    raise SystemExit(f"Gemini request failed: {exc}\nIf it says 503/high demand, wait a few minutes and run again.")

print("\nAnswer:")
print(response.text or "(Gemini returned an empty answer)")
print("\nSources:")
for i, hit in enumerate(hits, start=1):
    print(f"[{i}] {hit.title} (rrf={hit.vector_score:.4f}) {hit.url}")
