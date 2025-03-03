import sys
from pprint import pprint
from graph import app
from langchain.schema import Document



while True:
    user_input = input("Enter your question (or type 'exit' to quit): ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Exiting... Goodbye!")
        sys.exit()

    inputs = {"question": user_input}

    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f" Processing Node: '{key}'")
    
    documents = value.get("documents", [])
    if not documents:
        print("No relevant results found.")
        continue
    
    print("Result: ")

    if isinstance(documents, Document):
        print("Wikipedia Summary: ")
        pprint(documents.page_content)
    elif isinstance(documents, list):
        print("VectorStore Summary: ")
        pprint(documents[0].dict()['metadata']['description'])
    else:
        print("Unexpected output format. Here is the raw output:")
        pprint(documents)

    print("--------------------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------------------")
