"""
FastAPI server to expose the File Agent via the Model-Context Protocol (MCP).
"""
import json
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path to allow importing 'agent'
sys.path.append(str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from agent.agent_core import FileAgent


# --- Pydantic models for OpenAI-compatible API ---

class ChatMessage(BaseModel):
    """Schema for a single chat message."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Schema for the chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7


class ChatCompletionResponseChoice(BaseModel):
    """Schema for a choice in the response."""
    index: int
    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    """Schema for the full chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]


# --- FastAPI App and Agent Initialization ---

app = FastAPI(
    title="File Agent MCP Server",
    description="MCP-compatible server for the FileAgent.",
)

workspace_folder = Path(__file__).parent.parent / "workspace"
agent = FileAgent(base_path=str(workspace_folder))
print("FileAgent initialized and ready.")


# --- MCP Server Endpoints ---

@app.get("/", response_class=JSONResponse)
def get_manifest():
    """Serves the mcp_config.json manifest file."""
    manifest_path = Path(__file__).parent.parent / "mcp_config.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="mcp_config.json not found")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest):
    """Main chat endpoint, compatible with OpenAI's API."""
    print(f"Received request for model: {request.model}")

    user_query = next(
        (msg.content for msg in reversed(request.messages) if msg.role == "user"), None
    )

    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in request.")

    print(f"Query for agent: {user_query}")
    agent_response_dict = agent.plan(user_query)
    agent_output = agent_response_dict.get('output', "Agent did not provide an output.")
    print(f"Response from agent: {agent_output}")

    response_message = ChatMessage(role="assistant", content=agent_output)
    choice = ChatCompletionResponseChoice(index=0, message=response_message)
    return ChatCompletionResponse(model=request.model, choices=[choice])
