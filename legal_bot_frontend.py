import streamlit as st
from rag import RAGEngine
from llm import generate_response
import fitz  # PyMuPDF

rag = RAGEngine()

st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("📄 AI Legal Assistant")

# --- Upload Section ---
st.header("📤 Upload Legal Document")
uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.read()

    try:
        if file_name.endswith(".txt"):
            text = file_bytes.decode("utf-8", errors="ignore")
            chunks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            metadatas = [{"source": file_name}] * len(chunks)

        elif file_name.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            chunks, metadatas = [], []
            for i, page in enumerate(doc):
                page_text = page.get_text("text").strip()
                for para in page_text.split('\n\n'):
                    para = para.strip()
                    if len(para) > 50:
                        chunks.append(para)
                        metadatas.append({"source": file_name, "page": i + 1})

        else:
            st.error("Unsupported file type.")
            st.stop()

        if not chunks:
            st.warning("No valid text chunks found in file.")
            st.stop()

        if file_name in {m["source"] for m in rag.metadatas}:
            st.warning(f"'{file_name}' is already indexed.")
        else:
            before = len(rag.texts)
            rag.index.add(rag._normalize(rag.model.encode(chunks)).astype("float32"))
            rag.texts.extend(chunks)
            rag.metadatas.extend(metadatas)
            rag._persist()
            added = len(rag.texts) - before
            st.success(f"✅ Indexed {added} new chunks from '{file_name}'")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")

# --- Ask a Question ---
st.header("🔍 Ask a Legal Question")
query = st.text_input("Enter your legal question:")

if query:
    with st.spinner("Retrieving relevant legal context..."):
        results = rag.retrieve(query, k=5)
        context_chunks = [r["text"] for r in results]
        metadata_list = [r["metadata"] for r in results]

    with st.expander("📚 Show Top Matching Chunks"):
        for i, (text, meta) in enumerate(zip(context_chunks, metadata_list)):
            page_info = f" (Page {meta['page']})" if "page" in meta else ""
            st.markdown(f"**{i+1}.** `{meta.get('source', 'Unknown')}`{page_info}")
            st.code(text, language="markdown")

    with st.spinner("Generating answer..."):
        answer = generate_response(query, context_chunks)

    st.subheader("🤖 AI Answer")

    if context_chunks:
        avg_score = sum(r["score"] for r in results) / len(results)
        threshold_score = 0.3  # Tune this threshold based on your setup

        if avg_score >= threshold_score:
            st.markdown("📄 **Answer based on uploaded legal documents:**")
        else:
            st.markdown(
                "⚠️ *The retrieved content is weakly related; answer mainly from general knowledge.*"
            )
    else:
        st.markdown(
            "⚠️ *No matching content found in your documents. This answer is based on general knowledge.*"
        )

    st.write(answer)

# --- Delete Files Section ---
st.sidebar.header("🗑️ Manage Indexed Files")
file_sources = list({meta["source"] for meta in rag.metadatas})

if not file_sources:
    st.sidebar.info("No files indexed yet.")
else:
    to_delete = st.sidebar.selectbox("Select a file to delete:", file_sources)
    if st.sidebar.button("Delete File"):
        rag.delete_by_source(to_delete)
        st.sidebar.success(f"Deleted all chunks from '{to_delete}'")
        st.rerun()
