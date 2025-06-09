# AI File Management Agent

This project implements an advanced **tool-using AI agent**, built with **LangChain**, capable of intelligently and securely interacting with a local file system. The agent is exposed via both a Command-Line Interface (CLI) and a **Model-Context Protocol (MCP)** compliant server, making it ready for integration with modern editors like Cursor.

## ✨ Key Features

-   **🤖 Tool-Using Agent**: The agent can perform **CRUD** (Create, Read, Update, Delete) operations on files and answer complex questions based on the content of multiple files within a sandboxed workspace.
-   **🏗️ Modern Architecture**: Utilizes the modern LangChain stack with `AgentExecutor` and `create_openai_tools_agent`. Tools are robustly defined using `StructuredTool` and `Pydantic` schemas to ensure reliable argument handling.
-   **🛡️ Safety Guardrail**: The agent implements a "guardrail" that pre-emptively analyzes user requests. It actively declines to answer irrelevant or off-topic questions and explains why.
-   **⚡ 2-LLM Architecture**: To optimize for cost and latency, the agent uses a two-model architecture:
    -   A fast and lightweight model (**`gpt-3.5-turbo`**) for the initial classification of the query's topic.
    -   A powerful model (**`gpt-4o`**) for complex reasoning and tool use, engaged only when necessary.
-   **🖥️ CLI Interface**: An interactive command-line interface (`chat_interface/cli.py`) for easy testing and conversation with the agent.
-   **🌐 MCP Server**: A built-in MCP server (`server/mcp_server.py`) allows the agent to be used by compatible third-party clients like Cursor.
-   **✅ Comprehensive Test Suite**: The project includes a full `pytest` suite covering both the individual tools and the agent's core logic.

## 📂 Project Structure

```bash
/project-root
├── agent/
│   └── agent_core.py
├── chat_interface/
│   └── cli.py
├── server/
│   └── mcp_server.py
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   └── test_tools.py
├── tools/
│   └── tools.py
├── mcp_config.json
├── requirements.txt
└── README.md

## 🚀 Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/benedetto-moschella/file-management-agent.git
    cd project-root
    ```

2.  **Create and Activate the Conda Environment**
    This project uses Conda for environment management.
    ```bash
    # Create a new conda environment named 'file-agent-env' with Python 3.10
    conda create --name file-agent-env python=3.10 -y

    # Activate the environment
    conda activate file-agent-env
    ```

3.  **Install Dependencies**
    Once the environment is activated, install the required packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

The agent requires an API key for OpenAI models.

1.  Create a file named `.env` in the project's root directory.
2.  Add your API key to the `.env` file in the following format:
    ```
    OPENAI_API_KEY="sk-..."
    ```

## 🛠️ Usage

### 1. Interactive CLI

To chat with the agent directly in your terminal (make sure your conda environment is active):
```bash
python chat_interface/cli.py

**Example On-Topic Query:**
> `First, create a file named ingredients.txt and write "Guanciale, Eggs, Pecorino, Pepper" in it. Then, create a second file named recipe.txt and write "1. Fry the guanciale. 2. Combine eggs and cheese." inside it. Finally, tell me what to do after frying the guanciale.`

**Example Off-Topic Query:**
> `What is the tallest mountain in the world?`

### 2. MCP Server
To run the agent as a server for clients like Cursor:

```bash
uvicorn server.mcp_server:app --reload

The server will be available at http://localhost:8000.

### 3. Running Tests
To run the entire test suite:

```bash
pytest
