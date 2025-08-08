import os
import fitz  # PyMuPDF
from rag import KnowledgeGraph
from llm import extract_triples_from_text

# --- Config ---
PDF_PATH     = "docs/constitution.pdf"
CHUNK_MINLEN = 5   # skip tiny paragraphs

os.makedirs("storage", exist_ok=True)

kg = KnowledgeGraph()
doc = fitz.open(PDF_PATH)
base_name = os.path.basename(PDF_PATH)

for page_num, page in enumerate(doc, start=1):
    text = page.get_text("text").strip()
    if not text:
        continue

    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= CHUNK_MINLEN]

    for idx, para in enumerate(paras):
        para_id = f"{base_name}_p{page_num}_{idx}"
        
        # Extract triples from this paragraph
        triples = extract_triples_from_text(para)
        
        if not triples:
            # fallback to store paragraph as-is if no triples found
            kg.add_fact(
                subject=para_id,
                relation="has_paragraph",
                obj=para,
                source=base_name
            )
            continue

        for triple in triples:
            kg.add_fact(
                subject=triple["subject"],
                relation=triple["relation"],
                obj=triple["object"],
                source=f"{base_name}_p{page_num}"
            )

# Detect communities (Leiden)
kg.detect_communities()

# Save graph
kg.persist()
print("✅ Knowledge graph build complete.")
