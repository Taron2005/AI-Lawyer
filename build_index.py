import os
from rag_manager import get_rag_manager

# IMPORTANT: Create a 'docs' folder and place your .txt and .pdf files there.
# This script will automatically find and index them.
DOCS_DIRECTORY = "docs"

def main():
    """
    Initializes the RAG Manager and populates the index with all documents
    found in the DOCS_DIRECTORY.
    """
    print("🚀 Starting initial document indexing...")
    if not os.path.exists(DOCS_DIRECTORY):
        print(f"⚠️ Warning: '{DOCS_DIRECTORY}' folder not found. Please create it and add your documents.")
        return

    rag_manager = get_rag_manager()
    
    for filename in os.listdir(DOCS_DIRECTORY):
        file_path = os.path.join(DOCS_DIRECTORY, filename)
        if os.path.isfile(file_path) and (filename.endswith('.pdf') or filename.endswith('.txt')):
            print(f"\n--- Processing: {filename} ---")
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                rag_manager.add_document(file_content, filename)
                print(f"✅ Successfully processed and indexed '{filename}'.")
            except Exception as e:
                print(f"❌ Error processing file {filename}: {e}")

    print("\n🎉 Initial indexing complete.")
    print(f"Total vectors in index: {rag_manager.index.ntotal}")

if __name__ == "__main__":
    main()
