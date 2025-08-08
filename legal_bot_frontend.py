import os
import streamlit as st
from rag import KnowledgeGraph
from llm import generate_response
import fitz  # PyMuPDF
from speech_api import tts_long_text

def delete_audio_files():
    for file_path in st.session_state.audio_files:
        try:
            os.remove(file_path)
        except Exception:
            pass
    st.session_state.audio_files = []

kg = KnowledgeGraph()

if "audio_files" not in st.session_state:
    st.session_state.audio_files = []

st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("📄 AI Legal Assistant (Knowledge Graph)")

# — Ask a Question —
st.header("🔍 Ask a Legal Question")
query = st.text_input("Enter your legal question:")

if query:
    delete_audio_files()
    with st.spinner("Querying knowledge graph..."):
        facts = kg.query(query, k=5)
        context_chunks = [f"{f['subject']} {f['relation']} {f['object']}" for f in facts]
        sources = [f["source"] for f in facts]

    with st.expander("📚 Retrieved Triples"):
        for i, (ctx, src) in enumerate(zip(context_chunks, sources)):
            st.markdown(f"**{i+1}.** {ctx}  _(source: {src})_")

    with st.spinner("Generating answer..."):
        answer = generate_response(query, context_chunks)
        success, audio_files_or_err = tts_long_text(answer, base_filename="answer")

    st.subheader("🤖 AI Answer")
    st.write(answer)

    if success:
        st.session_state.audio_files = audio_files_or_err
        for path in audio_files_or_err:
            with open(path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
    else:
        st.error(audio_files_or_err)