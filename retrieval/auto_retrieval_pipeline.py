from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(**file**).resolve().parent.parent
PERSIST_DIRECTORY = str(BASE_DIR / "ChromaDB")

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

model = ChatGoogleGenerativeAI(
model="gemini-2.5-flash",
temperature=0
)

def get_vector_store(collection_name):
return Chroma(
collection_name=collection_name,
embedding_function=embedding_model,
persist_directory=PERSIST_DIRECTORY
)

def retrieve(query, collection_name, k=5):
vector_store = get_vector_store(collection_name)

```
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k,
        "fetch_k": 25,
        "lambda_mult": 0.5
    }
)

print(f"\nSearching '{collection_name}' collection...")

return retriever.invoke(query)
```

def generate_answer(query, documents):
if not documents:
return "I could not find any relevant information in the provided documents."

```
context = "\n\n".join(
    document.page_content
    for document in documents
)

prompt = ChatPromptTemplate.from_template(
    """
```

You are a document question-answering assistant.

Answer the user's question using only the provided context.

Do not use outside knowledge.

If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

```
chain = prompt | model | StrOutputParser()

return chain.invoke(
    {
        "context": context,
        "question": query
    }
)
```

def retrieve_and_answer(query, collection_name, k=5):
documents = retrieve(
query=query,
collection_name=collection_name,
k=k
)

```
answer = generate_answer(
    query=query,
    documents=documents
)

return answer, documents
```
