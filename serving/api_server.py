"""
KlavoraAI FastAPI Server for vLLM Inference

This server provides an OpenAI-compatible API for fine-tuned adapters.
Supports both Qwen3-4B and Phi-4-mini adapters via HuggingFace Hub.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Note: vLLM imports are lazy - only needed when server runs with GPU
# from vllm import LLM, SamplingParams


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[list[str]] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


class ModelInfo(BaseModel):
    model_name: str
    adapter_path: Optional[str]
    is_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and cleanup vLLM model on startup/shutdown."""
    # In a full deployment, you would initialize vLLM here:
    # llm = LLM(model=os.getenv("MODEL_NAME"), adapter_path=os.getenv("ADAPTER_PATH"))
    # app.state.llm = llm
    yield
    # Cleanup: app.state.llm = None


app = FastAPI(
    title="KlavoraAI Inference API",
    description="OpenAI-compatible API for contract and policy extraction adapters",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
async def list_models() -> dict[str, Any]:
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "StMark007/klavora-contract-qwen3-4b",
                "object": "model",
                "created": 1700000000,
                "owned_by": "StMark007",
            },
            {
                "id": "StMark007/klavora-policy-qwen3-4b",
                "object": "model",
                "created": 1700000000,
                "owned_by": "StMark007",
            },
            {
                "id": "StMark007/klavora-contract-phi4-mini",
                "object": "model",
                "created": 1700000000,
                "owned_by": "StMark007",
            },
            {
                "id": "StMark007/klavora-policy-phi4-mini",
                "object": "model",
                "created": 1700000000,
                "owned_by": "StMark007",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: CompletionRequest) -> dict[str, Any]:
    """
    OpenAI-compatible chat completions endpoint.

    For production deployment with vLLM, this would call:
    outputs = llm.generate([request.prompt], SamplingParams(...))
    """
    if request.stream:
        return StreamingResponse(
            _stream_response(request),
            media_type="text/event-stream",
        )

    # Placeholder response - replace with actual vLLM call in deployment
    import time

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or "StMark007/klavora-contract-qwen3-4b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"contract_type": "nda", "parties": ["Acme Corp", "Beta Inc"], "effective_date": "2024-01-01"}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": 20,
            "total_tokens": len(request.prompt.split()) + 20,
        },
    }


async def _stream_response(request: CompletionRequest) -> AsyncIterator[str]:
    """Generate streaming response."""
    import asyncio
    import json

    model_response = {
        "id": f"chatcmpl-stream-{id(request)}",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": request.model or "StMark007/klavora-contract-qwen3-4b",
    }

    sample_response = '{"contract_type": "nda", "parties": ["Acme Corp", "Beta Inc"]'
    for i, char in enumerate(sample_response):
        chunk = {
            **model_response,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": char},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)

    yield f"data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    """Legacy completions endpoint for compatibility."""
    import time

    model_name = request.model or "StMark007/klavora-contract-qwen3-4b"

    return CompletionResponse(
        id=f"cmpl-{int(time.time())}",
        created=int(time.time()),
        model=model_name,
        choices=[
            {
                "text": '{"contract_type": "nda", "parties": ["Acme Corp"]}',
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": 15,
            "total_tokens": len(request.prompt.split()) + 15,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
