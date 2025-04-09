from dotenv import load_dotenv

load_dotenv()
from graph.make_graph import app

if __name__=='__main__':
    print("Advanced RAG!")
    print(app.invoke({"question": "What is agent memory?"}))

# Agent memory refers to the capability of an agent to retain and recall information over extended periods.
# It includes short-term memory for in-context learning and long-term memory for storing vast amounts of information.
# Memory is a crucial component in autonomous agent systems powered by large language models like LLM.
