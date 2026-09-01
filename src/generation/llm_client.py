from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI

from src.config import settings
from src.core.exceptions import ConfigurationError, GenerationError


class BaseLLMClient(ABC):
    """Abstract interface for LLM text generation."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generates an answer string from system and user prompt."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class OpenAILLMClient(BaseLLMClient):
    """LLM client for OpenAI chat completion models (e.g. gpt-4o-mini)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self._model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise GenerationError("OPENAI_API_KEY is required for OpenAILLMClient.")
        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else "I don't know."
        except Exception as e:
            raise GenerationError(f"OpenAI completion failed: {str(e)}") from e

    @property
    def model_name(self) -> str:
        return self._model_name


class MockLLMClient(BaseLLMClient):
    """Deterministic, context-grounded mock LLM for offline testing and evaluation without API keys.
    Extracts answers directly from provided context chunks or responds with 'I don't know.'"""

    def __init__(self, model_name: str = "mock-llm-grounded-v1"):
        self._model_name = model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        # Extract context block from prompt
        context_match = re.search(r"Context:\s*---\s*(.*?)\s*---\s*Question:\s*(.*?)\s*Answer:", user_prompt, re.DOTALL)
        if not context_match:
            return "I don't know."

        context_text = context_match.group(1).strip()
        query_text = context_match.group(2).strip()

        if not context_text or context_text == "No relevant context found.":
            return "I don't know."

        # Extract meaningful query keywords (length >= 3, excluding stopwords)
        stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "the", "and", "for", "are", "is", "was", "were", "described", "specified", "under", "with", "from"}
        query_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text.lower()) if w not in stopwords and len(w) >= 3]

        if not query_tokens:
            return "I don't know."

        # Find matching sentences in the context
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context_text) if s.strip()]
        scored_sentences = []
        for s_clean in sentences:
            # Ignore chunk headers and source citations
            if s_clean.startswith("[CHUNK") or s_clean.startswith("Source:") or s_clean.startswith("---"):
                continue
            s_lower = s_clean.lower()
            
            # Weighted match: longer words and specific terms receive higher weights
            score = 0.0
            matched_terms = 0
            for tok in query_tokens:
                if tok in s_lower:
                    matched_terms += 1
                    # Give higher weight to action words / specific nouns
                    score += 1.0 + (len(tok) / 10.0)
            
            # Bonus if sentence contains key numbers or answer indicators
            if re.search(r"\b\d+\b", s_clean):
                score += 0.5

            if matched_terms > 0:
                scored_sentences.append((score, matched_terms, s_clean))

        if not scored_sentences:
            return "I don't know."

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_score, best_matches, best_sentence = scored_sentences[0]
        
        # Check threshold
        if best_matches < max(1, len(query_tokens) // 3):
            return "I don't know."

        # Return best matching sentence(s)
        top_sentences = [best_sentence]
        for score, matches, s_text in scored_sentences[1:]:
            if score >= best_score * 0.8 and s_text not in top_sentences:
                top_sentences.append(s_text)
                if len(top_sentences) >= 2:
                    break

        return " ".join(top_sentences)

    @property
    def model_name(self) -> str:
        return self._model_name


def get_llm_client(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMClient:
    """Factory to retrieve configured LLM client."""
    selected_provider = (provider or settings.LLM_PROVIDER).lower().strip()

    if selected_provider == "openai":
        key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not key or key == "your-openai-api-key-here":
            print("[Warning] OpenAI API key not configured. Falling back to MockLLMClient.")
            return MockLLMClient()
        return OpenAILLMClient(
            model_name=model_name or settings.LLM_MODEL,
            api_key=key,
        )

    elif selected_provider == "mock":
        return MockLLMClient()

    else:
        raise ConfigurationError(
            f"Unsupported LLM provider: '{selected_provider}'. Allowed: openai, mock"
        )
