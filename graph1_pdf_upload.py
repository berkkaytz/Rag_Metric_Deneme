# graph1_pdf_upload.py (LangGraph version)
from imports import *
import os

from services import short_hash_bytes, short_hash_text, iso_now, EMBED_MODEL, slugify

PROJECT_ID = os.getenv("PROJECT_ID", "default_project")

from typing import List, TypedDict
from langgraph.graph import StateGraph, END


# --------- State schema ---------
class G1State(TypedDict, total=False):
    pdf_path: str
    documents: List[Document]
    chunks: List[Document]
    doc_id: str
    result: str

# --------- Nodes ---------

def load_pdf_node(state: G1State) -> G1State:
    # Debug: show incoming path
    print("[Graph1] load_pdf_node: path=", state.get("pdf_path"))
    loader = PyPDFLoader(state["pdf_path"])
    docs = loader.load()
    # build deterministic doc_id from file bytes
    with open(state["pdf_path"], "rb") as f:
        fh = short_hash_bytes(f.read())
    state["doc_id"] = f"{slugify(os.path.basename(state['pdf_path']))}-{fh}"
    state["documents"] = docs
    print("[Graph1] load_pdf_node: docs=", len(docs))
    return state


def split_node(state: G1State) -> G1State:
    # Fail-safe: if documents missing, reload
    if "documents" not in state or state["documents"] is None:
        print("[Graph1] split_node: documents missing, reloading")
        loader = PyPDFLoader(state["pdf_path"])
        state["documents"] = loader.load()
    # Ensure doc_id exists (fallback if missing)
    if "doc_id" not in state or not state["doc_id"]:
        with open(state["pdf_path"], "rb") as f:
            fh = short_hash_bytes(f.read())
        state["doc_id"] = f"{slugify(os.path.basename(state['pdf_path']))}-{fh}"
        print("[Graph1] split_node: regenerated doc_id=", state["doc_id"])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHARS, chunk_overlap=SPLIT_OVERLAP
    )
    state["chunks"] = splitter.split_documents(state["documents"])
    print("[Graph1] split_node: chunks=", len(state["chunks"]))
    # Resolve project id locally to avoid NameError from globals/imports
    proj_id = os.getenv("PROJECT_ID", PROJECT_ID if 'PROJECT_ID' in globals() else "default_project")
    # enrich metadata per chunk
    for idx, ch in enumerate(state["chunks"]):
        page = int(ch.metadata.get("page") or 0)
        ch_text = ch.page_content
        ch.metadata.update({
            "doc_id": state["doc_id"],
            "chunk_id": f"{state['doc_id']}_p{page}_c{idx}",
            "page": page,
            "chunk_index": idx,
            "source": state["pdf_path"],
            "created_at": iso_now(),
            "split_method": "recursive",
            "overlap": SPLIT_OVERLAP,
            "token_count": len(ch_text.split()),
            "hash": short_hash_text(ch_text),
            "embedding_model": EMBED_MODEL,
            "project_id": proj_id,
        })
    return state


def enrich_and_store_node(state: G1State) -> G1State:
    # idempotent upsert into persistent vectordb
    db = vectordb()
    try:
        db.delete(where={"doc_id": state["doc_id"]})
    except Exception:
        pass
    db.add_documents(state["chunks"])
    db.persist()

    state["result"] = f"upserted={len(state['chunks'])} doc_id={state['doc_id']}"
    return state

# --------- Graph builder ---------

def build_graph1():
    g = StateGraph(G1State)
    g.add_node("load_pdf", load_pdf_node)
    g.add_node("split", split_node)
    g.add_node("store", enrich_and_store_node)
    g.set_entry_point("load_pdf")
    g.add_edge("load_pdf", "split")
    g.add_edge("split", "store")
    g.add_edge("store", END)
    return g.compile()

# Public API (kept for compatibility with app.py / main.py)

def run_graph1(pdf_path: str) -> str:
    app = build_graph1()
    out: G1State = app.invoke({"pdf_path": pdf_path})
    print("[Graph1]", out.get("result"))
    return out["doc_id"]

if __name__ == "__main__":
    run_graph1("uploaded.pdf")