"""
FastAPI server for AI Assistant Pro

Provides OpenAI-compatible API with streaming support
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncIterator
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from ai_assistant_pro.engine.model import AssistantEngine


# Global engine instance
engine: Optional[AssistantEngine] = None


class GenerationRequest(BaseModel):
    """Generation request model"""

    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stream: bool = False


class GenerationResponse(BaseModel):
    """Generation response model"""

    text: str
    tokens: int
    finish_reason: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model: str
    stats: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app"""
    global engine

    # Startup
    print("Initializing AI Assistant Pro engine...")
    engine = AssistantEngine(
        model_name="gpt2",  # Default model
        use_triton=True,
        use_fp8=True,
        enable_paged_attention=True,
    )
    print("✓ Engine ready")

    yield

    # Shutdown
    print("Shutting down...")


def create_app() -> FastAPI:
    """
    Create FastAPI application

    Returns:
        FastAPI app instance
    """
    app = FastAPI(
        title="AI Assistant Pro",
        description="High-performance AI assistant API optimized for NVIDIA Blackwell (SM120)",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint"""
        return {
            "message": "AI Assistant Pro API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint"""
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        return HealthResponse(
            status="healthy",
            model=engine.model_name,
            stats=engine.get_stats(),
        )

    @app.post("/generate", response_model=GenerationResponse)
    async def generate(request: GenerationRequest):
        """
        Generate text from prompt

        Args:
            request: Generation request

        Returns:
            Generated text response
        """
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        try:
            # Generate
            generated_text = engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stream=request.stream,
            )

            # Count tokens (approximate)
            tokens = len(engine.tokenizer.encode(generated_text))

            return GenerationResponse(
                text=generated_text,
                tokens=tokens,
                finish_reason="stop",
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions")
    async def openai_completions(request: GenerationRequest):
        """
        OpenAI-compatible completions endpoint

        Provides compatibility with OpenAI API clients
        """
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        try:
            generated_text = engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )

            return {
                "id": "cmpl-" + str(hash(generated_text))[:8],
                "object": "text_completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": engine.model_name,
                "choices": [
                    {
                        "text": generated_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    async def get_stats():
        """Get engine statistics"""
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        return engine.get_stats()

    return app


def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str = "gpt2",
    use_triton: bool = True,
    use_fp8: bool = True,
):
    """
    Start serving API

    Args:
        host: Host to bind to
        port: Port to bind to
        model_name: Model to load
        use_triton: Enable Triton kernels
        use_fp8: Enable FP8 quantization

    Example:
        >>> from ai_assistant_pro.serving import serve
        >>> serve(model_name="meta-llama/Llama-3.1-8B", port=8000)
    """
    # Initialize engine with custom settings
    global engine
    engine = AssistantEngine(
        model_name=model_name,
        use_triton=use_triton,
        use_fp8=use_fp8,
        enable_paged_attention=True,
    )

    # Create app
    app = create_app()

    # Run server
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
