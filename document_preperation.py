import os
import cassio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

#  Load environment variables
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

#  Initialize AstraDB connection
if not ASTRA_DB_APPLICATION_TOKEN or not ASTRA_DB_ID:
    raise ValueError("Missing Astra DB credentials. Check your .env file.")

try:
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    print(" Astra DB connection successful!")
except Exception as e:
    print(f" Astra DB connection failed: {e}")

#  Define the Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

#  Check if Data Already Exists
existing_docs = astra_vector_store.similarity_search("test", k=1)  # Query for any existing record

if existing_docs:
    print("Data source is prepared......")
else:
    print("No data found. Inserting new documents...")

    #  URLs to scrape
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    #  Document loading
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    #  Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)

    #  Insert documents only if needed
    astra_vector_store.add_documents(doc_splits)
    print(f" Inserted {len(doc_splits)} document chunks into AstraDB.")

#  Create an index & retriever
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever = astra_vector_store.as_retriever()
