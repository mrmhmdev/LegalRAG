import os
import logging
from openai import OpenAI
from retrieve import Retriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(question, context):
    return f"""Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Answer:"""


def answer_query(question, top_k=3):
    try:
        logger.info(f"Processing query: '{question}' with top_k={top_k}")
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY not configured")
        
        try:
            retriever = Retriever()
            logger.info("Retriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise
        
        try:
            retrieved_chunks = retriever.retrieve(question, top_k=top_k)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        except Exception as e:
            logger.error(f"Retrieval step failed: {str(e)}")
            raise
        
        context = "\n\n".join(retrieved_chunks)
        prompt = build_prompt(question, context)
        
        try:
            logger.info("Sending request to OpenAI API...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            logger.info("Received response from OpenAI API")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    except Exception as e:
        logger.critical(f"Answer generation pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        question = "What are the key points?"
        answer = answer_query(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
    except Exception as e:
        logger.critical(f"Main execution failed: {str(e)}")
