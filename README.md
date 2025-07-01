# AI File Management Agent

This project implements an advanced **tool-using AI agent**, built with **LangChain**, capable of intelligently and securely interacting with a local file system. The agent is exposed via both a Command-Line Interface (CLI) and a **Model-Context Protocol (MCP)** compliant server, making it ready for integration with modern editors like Cursor.

## ✨ Key Features

-   **🤖 Tool-Using Agent**: The agent can perform CRUD operations on files and perform semantic searches to answer complex questions based on file content within a sandboxed workspace.
-   **🧠 RAG Pipeline**: The agent uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions. Files are automatically chunked, vectorized using `Sentence-Transformers`, and indexed in a `FAISS` vector store for efficient and accurate semantic search.
-   **🏗️ Modern Architecture**: Utilizes the modern LangChain stack with `AgentExecutor` and `create_openai_tools_agent`. Tools are robustly defined using `StructuredTool` and `Pydantic` schemas to ensure reliable argument handling.
-   **🛡️ Safety Guardrail**: The agent implements a "guardrail" that pre-emptively analyzes user requests. It actively declines to answer irrelevant or off-topic questions and explains why.
-   **⚡ 2-LLM Architecture**: To optimize for cost and latency, the agent uses a two-model architecture:
    -   A fast and lightweight model (**`gpt-3.5-turbo`**) for the initial classification of the query's topic.
    -   A powerful model (**`gpt-4o`**) for complex reasoning and tool use, engaged only when necessary.
-   **🐳 Production-Ready Deployment**: The project is fully containerized with **Docker**, allowing for easy and reproducible deployment.
-   **✅ Comprehensive Test & Evaluation Suite**: The project includes a full `pytest` suite for unit testing and a dedicated script (`evaluate.py`) for performance evaluation.

## 📂 Project Structure

```bash
/project-root
├── agent/
│   └── agent_core.py
├── chat_interface/
│   └── cli.py
├── rag/
│   └── vector_store_manager.py
├── server/
│   └── mcp_server.py
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   └── test_tools.py
├── tools/
│   └── tools.py
├── workspace/
├── .dockerignore
├── .env
├── .gitignore
├── Dockerfile
├── evaluate.py
├── evaluation_dataset.jsonl
├── mcp_config.json
├── requirements.txt
└── README.md
```

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
```
**Example On-Topic Query:**
> `First, create a file named ingredients.txt and write "Guanciale, Eggs, Pecorino, Pepper" in it. Then, create a second file named recipe.txt and write "1. Fry the guanciale. 2. Combine eggs and cheese." inside it. Finally, tell me what to do after frying the guanciale.`

**Example Off-Topic Query:**
> `What is the tallest mountain in the world?`

**Example RAG Query:**
> `First, create a file named 'ai_risks.txt' with the content "One of the main dangers of AI is algorithmic bias." Then ask: "What are the dangers related to artificial intelligence?"`

### 2. MCP Server
To run the agent as a server for clients like Cursor:

```bash
uvicorn server.mcp_server:app --reload
```
The server will be available at http://localhost:8000.

### 3. Running with Docker
To build and run the project in a container (requires Docker Desktop to be running):

```bash
# 1. Build the Docker image
docker build -t file-agent .

# 2. Run the container
docker run -p 8000:8000 -v ./workspace:/app/workspace --env-file .env file-agent
```

### 4. Running Tests & Evaluation
To run the entire test suite:

```bash
pytest
```

To evaluate the agent's performance on a predefined set of questions:

```bash
python evaluate.py
```