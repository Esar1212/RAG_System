from langchain_groq import ChatGroq
from app.retrieve import RAGRetriever as RAG
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# # The client gets the API key from the environment variable `GROQ_API_KEY`.
# groq_api_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(groq_api_key=groq_api_key,model_name = "llama-3.3-70b-versatile",temperature=0.1,max_tokens=1024)



# def rag(doc_type, query,retriever,llm,top_k=3):
#     ## retrieving the context
#     results = retriever.retrieve(doc_type,query,top_k=top_k)
#     context = [result['content'] for result in results] if results else ""
#     # print(context)
#     if not context:
#         return "No relevant documents found to answer the requested query."
    
#     ## System prompt for the LLM
#     system_prompt = f"""You are a helpful assistant for answering queries with respect to a given context.
#     Context: {{context}}
#     Query: {{query}}
#     Answer the query based only on the provided context. Please try to be as concise as possible while answering the query. Do not hallucinate or provide any information which is not present in the provided context.
#       NB: If you encounter \n\n or \n in the context, treat them as newline escape sequence, please go to a different line and do not include these characters in the answer."""
    
#     response = llm.invoke([system_prompt.format(context=context, query=query)])
#     return response.content

# query = "Explain adjacency list representation of a graph"
# answer = rag('hello_docs',query, RAG, llm,top_k=10)
# answer

import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq


class RGenerator:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        # self.groq_api_key = os.getenv("GROQ_API_KEY")  # Please use this statement under development environment when you have the .env file in the directory
          self.groq_api_key = st.secrets["GROQ_API_KEY"]
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[str]
    ) -> str:
        """Generate answer using retrieved context"""

        if not retrieved_docs:
            return "No relevant documents found to answer the query."

        context = "\n\n".join(retrieved_docs)

        prompt = f"""
You are a helpful assistant for answering queries based ONLY on the given context.

Context:
{context}

Query:
{query}

Instructions:
- Answer only using the provided context
- Be concise and accurate
- Do NOT hallucinate
- Treat \\n and \\n\\n as line breaks

Answer:
"""

        response = self.llm.invoke(prompt)
        return response.content
