import logging
import streamlit as st
from answer import answer_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📚 RAG Question Answering System")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of retrieved chunks", 1, 10, 7)
    
    st.markdown("""
    ### How it works:
    1. **Ingest**: PDFs are chunked and embedded
    2. **Retrieve**: Query finds similar chunks via FAISS
    3. **Generate**: LLM answers using retrieved context
    """)

question = st.text_input("Ask a question:", placeholder="What is...")

if question:
    with st.spinner("Retrieving and generating answer..."):
        try:
            logger.info(f"Streamlit: Processing question '{question}'")
            answer = answer_query(question, top_k=top_k)
            logger.info("Streamlit: Answer generated successfully")
            st.success("Answer generated!")
            st.write(answer)
        except ValueError as e:
            logger.error(f"Streamlit validation error: {str(e)}")
            st.error(f"Configuration error: {str(e)}")
        except Exception as e:
            logger.error(f"Streamlit error: {str(e)}")
            st.error(f"Error: {str(e)}")
