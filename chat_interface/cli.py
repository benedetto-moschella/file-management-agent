"""
A simple command-line interface to chat with the File Agent.
"""
import sys
from pathlib import Path

# Add project root to path to allow importing 'agent'
sys.path.append(str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from agent.agent_core import FileAgent


def main():
    """
    Main loop for the CLI chat.
    """
    workspace_folder = Path(__file__).parent.parent / "workspace"
    agent = FileAgent(base_path=str(workspace_folder))

    print("Welcome to FileAgent CLI. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            response = agent.plan(user_input)
            print(f"Agent: {response.get('output', 'No response output.')}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
    