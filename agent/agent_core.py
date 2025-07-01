"""Core logic for the File Agent."""
# Standard library imports
from typing import List

# Third-party imports
from dotenv import load_dotenv
from langchain import hub
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# Local application imports
from tools.tools import FileTools

load_dotenv()


# pylint: disable=too-few-public-methods
class WriteFileInput(BaseModel):
    """Input schema for the write_file tool."""
    filename: str = Field(description="The name of the file to write to.")
    content: str = Field(description="The content to write into the file.")


# pylint: disable=too-few-public-methods
class ReadFileInput(BaseModel):
    """Input schema for the read_file tool."""
    filename: str = Field(description="The name of the file to read.")


# pylint: disable=too-few-public-methods
class DeleteFileInput(BaseModel):
    """Input schema for the delete_file tool."""
    filename: str = Field(description="The name of the file to delete.")


# pylint: disable=too-few-public-methods
class AnswerAboutFilesInput(BaseModel):
    """Input schema for the answer_question_about_files tool."""
    query: str = Field(description="The question to answer based on the content of the files.")


class FileAgent:
    """A file management agent that can perform CRUD and query operations."""

    def __init__(self, base_path: str, llm_model: str = "gpt-4o", temperature: float = 0.0):
        """Initializes the agent with its tools and models."""
        self.file_tools = FileTools(base_path)
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.classifier_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = hub.pull("hwchase17/openai-tools-agent")
        tools = self._build_tool_list()
        agent_runnable = create_openai_tools_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)

    def _build_tool_list(self) -> List[StructuredTool]:
        """Builds the list of structured tools available to the agent."""
        return [
            StructuredTool.from_function(
                func=self.file_tools.list_files,
                name="list_files",
                description="List all files in the workspace."
            ),
            StructuredTool.from_function(
                func=self.file_tools.read_file,
                name="read_file",
                description="Read the content of a file.",
                args_schema=ReadFileInput
            ),
            StructuredTool.from_function(
                func=self.file_tools.write_file,
                name="write_file",
                description="Write content to a file.",
                args_schema=WriteFileInput
            ),
            StructuredTool.from_function(
                func=self.file_tools.delete_file,
                name="delete_file",
                description="Delete a file.",
                args_schema=DeleteFileInput
            ),
            StructuredTool.from_function(
                func=self.file_tools.answer_question_about_files,
                name="answer_question_about_files",
                description=(
                    "Use this tool as the primary way to answer user questions. "
                    "It performs a semantic search over the content of all files in the workspace "
                    "to find the most relevant information for the given query."
                ),
                args_schema=AnswerAboutFilesInput
            ),
        ]

    def _is_query_on_topic(self, user_query: str) -> bool:
        """Uses a classifier LLM to determine if the query is on-topic."""
        system_prompt = (
            'You are a precise text classifier. Your only function is to determine if a query '
            'is "ON-TOPIC" or "OFF-TOPIC".\n'
            '"ON-TOPIC" means the query is strictly about file management (list, read, '
            'write, delete) OR it is a question that can likely be answered using the '
            'content of the files in the workspace.\n'
            'Any other query, like a general knowledge question (e.g., "what is the capital '
            'of France?"), is "OFF-TOPIC".\n'
            'You must respond with ONLY the word "ON-TOPIC" or "OFF-TOPIC".'
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        response = self.classifier_llm.invoke(messages)
        classification = response.content.strip().upper()
        print(
            f"--- Topic Classification (with {self.classifier_llm.model_name}): "
            f"{classification} ---"
        )
        return classification == "ON-TOPIC"

    def plan(self, user_query: str) -> dict:
        """
        Executes the agent's plan, using an optimized two-step verification.
        """
        # --- LOGICA OTTIMIZZATA ---
        if self._is_query_on_topic(user_query):
            # Se il classificatore è sicuro, procediamo con l'AgentExecutor.
            print("Classifier result: ON-TOPIC. Proceeding to main agent executor.")
            inputs = {"input": user_query}
            result = self.agent_executor.invoke(inputs)
            return {"output": result["output"]}

        # Se il classificatore dice OFF-TOPIC, eseguiamo il fallback RAG.
        print("Classifier result: OFF-TOPIC. Performing fallback similarity search...")
        search_context = self.file_tools.answer_question_about_files(query=user_query)

        if "No relevant information found" in search_context:
            # Se anche il RAG non trova nulla, la domanda è davvero off-topic.
            print("Fallback search confirmed: No relevant documents. Declining.")
            refusal_message = (
                "I am a file management assistant. I cannot answer off-topic "
                "or general knowledge questions."
            )
            return {"output": refusal_message}

        # Se il RAG ha trovato contesto, bypassiamo l'AgentExecutor e generiamo la risposta!
        print("Fallback search found relevant context. Generating answer directly...")
        final_prompt = (
            "You are a helpful assistant. Please answer the user's question based "
            "only on the following context provided.\n\n"
            f"CONTEXT:\n{search_context}\n\n"
            f"USER'S QUESTION: {user_query}"
        )

        # Invochiamo l'LLM principale solo per il compito di generazione finale.
        final_response = self.llm.invoke(final_prompt)

        return {"output": final_response.content}
    