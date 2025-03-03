from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from document_preperation import retriever 
from langgraph.graph import END, StateGraph, START
from typing import Literal
import os

#  Load API Key for Groq
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(" Missing GROQ_API_KEY in environment variables!")

#  Define the routing model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ..., description="Choose either Wikipedia or Vectorstore for routing."
    )

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
structured_llm_router = llm.with_structured_output(RouteQuery)

#  Define Routing Prompt
system_message = """You are an expert at routing a user question to a vectorstore or Wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki_search."""

route_prompt = ChatPromptTemplate.from_messages(
    [("system", system_message), ("human", "{question}")]
)

question_router = route_prompt | structured_llm_router

#  Wikipedia Search Setup
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

#  Graph State Definition
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

#  Retrieve Function
def retrieve(state):
    """
    Retrieve documents from AstraDB Vector Store.
    """
    question = state["question"]
    documents = retriever.invoke(question) 
    return {"documents": documents, "question": question}

#  Wikipedia Search Function
def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state["question"]

    # Wiki search
    docs = wiki.invoke({"query": question})
    #print(docs["summary"])
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}

#  Routing Function
def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        return "wiki_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"

workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)

# Build routing edges
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

app = workflow.compile()
