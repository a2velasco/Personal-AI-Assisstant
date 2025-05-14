from langchain_core.messages import HumanMessage # langchain high level framework which allows to build AI apps
from langchain_openai import ChatOpenAI 
from langchain.tools import tool 
from langgraph.prebuilt import create_react_agent # allows to build AI agents
from dotenv import load_dotenv # loads environment variables from .env file 

load_dotenv() # Looks into .env file for OpenAI API key

@tool
def calculator(a: float, b: float) -> str:
    # doc string which describes the tool so the agent knows when to use it
    """Useful for performing basic arithmeric calculations with numbers""" 
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"

@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    print("Tool has been called.")
    return f"Hello {name}, I hope you are well today"

def main():
    model = ChatOpenAI(temperature=0) # Initialize the OpenAI model with a temperature of 0 for deterministic output

    tools = [calculator, say_hello] # will fill with tools later
    agent_executor = create_react_agent(model, tools) # authomatically creates an agent executor with the model and tools
    
    print("Hello boss! I am your AI assistant. How can I help you today? Type 'quit' to exit.") # greeting message
    print("I can perform calculations or just chat with you.")

    while True: 
        user_input = input("\nYou: ").strip() # removes white spaces from the input

        if user_input == "quit":
            break # exit the loop if user types 'quit'

        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}): # sends the user input to the agent executor
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print() # print a new line after the response

if __name__ == "__main__":
    main() 