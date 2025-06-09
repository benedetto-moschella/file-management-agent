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
                description="Answer questions based on the content of files.",
                args_schema=AnswerAboutFilesInput
            ),
        ]

    def _is_query_on_topic(self, user_query: str) -> bool:
        """Uses a classifier LLM to determine if the query is on-topic."""
        system_prompt = (
            'You are a precise text classifier. Your only function is to determine if a query '
            'is "ON-TOPIC" or "OFF-TOPIC".\n'
            '"ON-TOPIC" means the query is strictly about file management (list, read, '
            'write, delete, ask about file content).\n'
            'Any other query is "OFF-TOPIC".\n'
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
        Executes the agent's plan, including a pre-check for topic relevance.
        """
        if not self._is_query_on_topic(user_query):
            refusal_message = (
                "I am a file management assistant. I cannot answer off-topic "
                "or general knowledge questions."
            )
            return {"output": refusal_message}

        inputs = {"input": user_query}
        result = self.agent_executor.invoke(inputs)
        return {"output": result["output"]}
    