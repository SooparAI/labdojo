#!/usr/bin/env python3
"""
Lab Dojo v0.1.2 - Serverless Inference Server
Runs vLLM with OpenAI-compatible API on Vast.ai Serverless
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('labdojo-inference')

# Environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct-AWQ")
PORT = int(os.environ.get("PORT", 8000))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 32768))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.95))

# FastAPI app
app = FastAPI(title="Lab Dojo Inference API", version="1.0")

# Global vLLM engine (lazy loaded)
llm_engine = None


def get_llm_engine():
    """Lazy load vLLM engine"""
    global llm_engine
    if llm_engine is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        from vllm import LLM
        llm_engine = LLM(
            model=MODEL_NAME,
            quantization="awq",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")
    return llm_engine


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "ready": llm_engine is not None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Lab Dojo Inference API",
        "model": MODEL_NAME,
        "version": "1.0"
    }


@app.post("/v1/completions")
async def completions(request: InferenceRequest):
    """OpenAI-compatible completions endpoint"""
    try:
        llm = get_llm_engine()
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        
        logger.info(f"Inference request: {len(request.prompt)} chars")
        
        outputs = llm.generate([request.prompt], sampling_params)
        output = outputs[0]
        
        return {
            "text": output.outputs[0].text,
            "tokens": len(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason,
            "model": MODEL_NAME
        }
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: Dict[str, Any]):
    """OpenAI-compatible chat completions endpoint"""
    try:
        llm = get_llm_engine()
        
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Build prompt from messages
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.9),
            max_tokens=request.get("max_tokens", 2048),
            stop=["<|im_end|>"]
        )
        
        logger.info(f"Chat request: {len(messages)} messages")
        
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        return {
            "id": f"chatcmpl-{hash(prompt)}",
            "object": "chat.completion",
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output.outputs[0].text
                },
                "finish_reason": output.outputs[0].finish_reason
            }],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            }
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Preload model on startup"""
    logger.info("Lab Dojo Inference Server starting...")
    logger.info(f"Model: {MODEL_NAME}")
    try:
        get_llm_engine()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


if __name__ == "__main__":
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
