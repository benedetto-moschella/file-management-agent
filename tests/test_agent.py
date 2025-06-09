"""
Tests for the FileAgent's logic, including classification and tool use cycles.
"""
# pylint: disable=redefined-outer-name, too-few-public-methods
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.agents import AgentAction, AgentFinish

# Adds the project root to the path to allow importing 'agent'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from agent.agent_core import FileAgent


class DummyClassifierLLM:
    """Mocks the classifier LLM."""
    def __init__(self, responses: list, model_name: str = "dummy-classifier"):
        self._responses = responses
        self.call_count = 0
        self.model_name = model_name

    def invoke(self, prompt: list, **kwargs):
        """Mocks the invoke call."""
        # pylint: disable=unused-argument
        if self.call_count < len(self._responses):
            response = self._responses[self.call_count]
            self.call_count += 1
            return AIMessage(content=response)
        return AIMessage(content="Fallback response.")


class DummyAgentRunnable:
    """Mocks the complete agent runnable (LLM + output parser)."""
    def __init__(self, responses: list):
        self._responses = responses
        self.call_count = 0
        self.input_keys = ["input"]
        self.return_values = ["output"]

    def plan(self, intermediate_steps: list, **kwargs):
        """Mocks the plan method called by the AgentExecutor."""
        # pylint: disable=unused-argument
        if self.call_count < len(self._responses):
            response = self._responses[self.call_count]
            self.call_count += 1
            return response
        return AgentFinish(return_values={"output": "Fallback response."}, log="Fallback.")

    def tool_run_logging_kwargs(self):
        """Mocks the logging method required by the AgentExecutor."""
        return {}


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> str:
    """Creates a temporary workspace for agent tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


def test_agent_declines_off_topic_query(tmp_workspace: str):
    """Tests that the agent correctly declines off-topic questions."""
    fake_classifier_llm = DummyClassifierLLM(responses=["OFF-TOPIC"])
    agent = FileAgent(base_path=tmp_workspace)
    agent.classifier_llm = fake_classifier_llm
    result = agent.plan("What is the capital of France?")
    assert "I cannot answer off-topic" in result['output']


def test_agent_writes_file_and_answers(tmp_workspace: str):
    """Tests a full agent cycle of using a tool and providing a final answer."""
    fake_classifier_llm = DummyClassifierLLM(responses=["ON-TOPIC"])

    file_to_write = "test.txt"
    content_to_write = "hello"

    action_step = [AgentAction(
        tool="write_file",
        tool_input={"filename": file_to_write, "content": content_to_write},
        log=f"Invoking `write_file` for {file_to_write}"
    )]
    finish_step = AgentFinish(
        return_values={"output": f"Successfully wrote to {file_to_write}."},
        log="Final answer."
    )

    fake_agent_runnable = DummyAgentRunnable(responses=[action_step, finish_step])
    agent = FileAgent(base_path=tmp_workspace)
    agent.classifier_llm = fake_classifier_llm
    agent.agent_executor.agent = fake_agent_runnable

    result = agent.plan(f"Write '{content_to_write}' to the file '{file_to_write}'.")

    assert (Path(tmp_workspace) / file_to_write).read_text() == content_to_write
    assert "Successfully wrote" in result['output']
