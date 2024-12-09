import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# App title
st.title("RAG Ebook Librarian with LlamaIndex")

# Sidebar for user configuration
with st.sidebar:
    st.header("Configuration")
    library_dir = st.text_input("Library Directory", "./test/")
    model_name = st.text_input("Embedding Model", "BAAI/bge-small-en-v1.5")
    llama_model = st.text_input("LLM Model", "llama2")
    query_timeout = st.number_input("Query Timeout (s)", value=40.0, step=0.1)

# Initialize session state
if "documents" not in st.session_state:
    st.session_state["documents"] = None

if "index" not in st.session_state:
    st.session_state["index"] = None

# Load the library
st.header("1. Load Library")
if st.button("Load Library"):
    try:
        loader = SimpleDirectoryReader(
            input_dir=library_dir,
            recursive=True,
            required_exts=[".epub"],
        )
        documents = loader.load_data()
        st.session_state["documents"] = documents  # Save documents to session state
        st.success(f"Loaded {len(documents)} documents.")
    except Exception as e:
        st.error(f"Error loading library: {e}")

# Index the documents
st.header("2. Index Documents")
if st.button("Index Documents"):
    try:
        if st.session_state["documents"] is None:
            st.warning("No documents loaded. Please load the library first.")
        else:
            embedding_model = HuggingFaceEmbedding(model_name=model_name)
            index = VectorStoreIndex.from_documents(
                st.session_state["documents"],
                embed_model=embedding_model,
            )
            st.session_state["index"] = index  # Save index to session state
            st.success("Indexing completed successfully!")
    except Exception as e:
        st.error(f"Error indexing documents: {e}")

# Query the library
st.header("3. Query Library")
query = st.text_input("Enter your query", placeholder="Ask about your library")
if st.button("Run Query"):
    try:
        if st.session_state["index"] is None:
            st.warning("No index available. Please index the documents first.")
        elif not query:
            st.warning("Please enter a query.")
        else:
            llama = Ollama(
                model=llama_model,
                request_timeout=query_timeout,
            )
            query_engine = st.session_state["index"].as_query_engine(llm=llama)
            response = query_engine.query(query)
            st.subheader("Query Response")
            st.text(response)
    except Exception as e:
        st.error(f"Error running query: {e}")

# Footer
st.markdown("---")
st.markdown(
    "Created with 💡 by [Jonathan Jin](https://huggingface.co/jinnovation), adapted for Streamlit by OpenAI."
)
