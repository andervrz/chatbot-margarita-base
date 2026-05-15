"""
Cliente LLM genérico mediante LiteLLM.
Permite cambiar de proveedor (Groq, OpenAI, Anthropic, etc.) sin tocar código de negocio.
"""

import structlog
from litellm import acompletion, aembedding

from src.config import settings
from src.domain.exceptions import LLMError

logger = structlog.get_logger()


class LLMClient:
    """
    Wrapper async sobre LiteLLM.
    - Chat: Groq (Llama 3.3 70B) por defecto.
    - Embeddings: OpenAI text-embedding-3-small (Groq no tiene embeddings nativos aún).
    """

    def __init__(self) -> None:
        self.chat_model = settings.llm_model
        self.embedding_model = settings.embedding_model
        self.default_temperature = settings.llm_temperature
        self.default_max_tokens = settings.llm_max_tokens

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Envía mensajes al LLM y retorna el contenido textual.
        messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
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
        Usa LiteLLM routing: 'openai/text-embedding-3-small' u otro modelo configurado.
        """
        try:
            # LiteLLM detecta el proveedor por el prefijo del modelo
            api_key = (
                if self.embedding_model.startswith("huggingface"):
                    api_key = settings.hf_api_key
                elif self.embedding_model.startswith("openai")
                    api_key = settings.openai_api_key
            )

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
