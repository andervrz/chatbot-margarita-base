"""
Cliente LLM genérico.
Chat vía LiteLLM + Groq. Embeddings vía LiteLLM (OpenAI) o HTTP directo (HF).
"""

import structlog
from litellm import acompletion, aembedding
import httpx

from src.config import settings
from src.domain.exceptions import LLMError

logger = structlog.get_logger()


class LLMClient:
    def __init__(self) -> None:
        self.chat_model = settings.llm_model
        self.embedding_model = settings.embedding_model
        self.default_temperature = settings.llm_temperature
        self.default_max_tokens = settings.llm_max_tokens
        # Exponer dimensión para que DB y tests no asuman
        self.embedding_dim = settings.embedding_dimensions

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.default_temperature
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens

        try:
            response = await acompletion(
                model=self.chat_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                api_key=settings.groq_api_key,
            )
            content = response.choices[0].message.content
            usage = response.usage.total_tokens if response.usage else None
            logger.debug(
                "llm_chat_success",
                model=self.chat_model,
                tokens=usage,
                max_tokens=max_tok,
            )
            return content.strip()
        except Exception as exc:
            logger.error("llm_chat_error", error=str(exc), model=self.chat_model)
            raise LLMError(f"Fallo en chat LLM: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        """
        Genera embedding del texto.
        - Si empieza con 'huggingface/': llama directo a HF Inference API vía httpx.
        - Si empieza con 'openai/': usa LiteLLM.
        """
        try:
            if self.embedding_model.startswith("huggingface/"):
                return await self._embed_hf_direct(text)
            
            # Fallback LiteLLM (OpenAI u otros)
            api_key = None
            if self.embedding_model.startswith("openai"):
                api_key = settings.openai_api_key

            response = await aembedding(
                model=self.embedding_model,
                input=text,
                api_key=api_key,
            )
            embedding: list[float] = response.data[0]["embedding"]
            logger.debug(
                "llm_embed_success",
                model=self.embedding_model,
                dimensions=len(embedding),
            )
            return embedding

        except Exception as exc:
            logger.error("llm_embed_error", error=str(exc), model=self.embedding_model)
            raise LLMError(f"Fallo en embedding: {exc}") from exc

    async def _embed_hf_direct(self, text: str) -> list[float]:
        """
        Llama directamente a Hugging Face Inference API vía router.
        Endpoint: router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction
        
        Nota: HF Inference API gratuita tiene cold start de 20-40s en el primer request.
        El timeout de 30s puede no ser suficiente para el primer embedding; en ese caso
        el llamador debe manejar el retry o usar warm_cache.py para "despertar" el modelo.
        """
        if not settings.hf_api_key:
            raise LLMError(
                "HF_API_KEY no configurada pero embedding_model usa huggingface/. "
                "Configura HF_API_KEY en .env o cambia el modelo de embeddings."
            )

        model_id = self.embedding_model.replace("huggingface/", "")
        url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{model_id}/pipeline/feature-extraction"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {settings.hf_api_key}",
                    "Content-Type": "application/json",
                },
                json={"inputs": text},
            )

            if response.status_code != 200:
                raise LLMError(
                    f"HF API error {response.status_code}: {response.text}"
                )

            data = response.json()

            # HF retorna [[vec]] para batch de 1
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    embedding = data[0]
                else:
                    embedding = data
            else:
                raise LLMError(f"Formato inesperado de HF: {data}")

            logger.debug(
                "hf_embed_success",
                model=model_id,
                dimensions=len(embedding),
            )
            return embedding
