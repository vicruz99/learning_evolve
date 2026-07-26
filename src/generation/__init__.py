"""LLM generation backends for the ICL harness."""
from generation.vllm_client import GenResult, VLLMClient

__all__ = ["GenResult", "VLLMClient"]
