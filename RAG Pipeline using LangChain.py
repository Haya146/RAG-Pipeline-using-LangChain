#dependencies pip install langchain chromadb openai tiktoken

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1️ Load Documents
loader = PyPDFLoader("data/company_profile.pdf")
documents = loader.load()

# 2️ Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 3️ Create Embeddings & Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4️ Create Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5️ Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 6️ Create RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7️ Ask a Question
query = "What services does this company provide?"
response = qa_chain({"query": query})

print("Answer:", response["result"])
print("\nSources:", [doc.metadata.get("source") for doc in response["source_documents"]])
