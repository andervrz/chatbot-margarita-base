"""
Enriquecimiento de prompts.
Toma datos reales de la base de datos y los inyecta en templates Jinja2.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


class PromptEnricher:
    """
    Construye prompts listos para el LLM usando templates Jinja2.
    Los prompts viven en /prompts (raíz del proyecto), no dentro del paquete src.
    """

    def __init__(self) -> None:
        # Subir dos niveles desde src/application/ hasta la raíz del proyecto
        prompts_dir = Path(__file__).resolve().parent.parent.parent / "prompts"
        self.env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=select_autoescape(),
        )

    def build_search_context(
        self,
        properties: list,
        user_query: str,
    ) -> str:
        template = self.env.get_template("search_results.txt.j2")
        return template.render(
            user_query=user_query,
            properties=properties,
            count=len(properties),
        )

    def build_vector_context(
        self,
        properties: list,
        user_query: str,
    ) -> str:
        template = self.env.get_template("vector_results.txt.j2")
        return template.render(
            user_query=user_query,
            properties=properties,
            count=len(properties),
        )

    def build_system_prompt(self) -> str:
        template = self.env.get_template("system_v1.txt")
        return template.render()

    def build_fallback_message(self) -> str:
        template = self.env.get_template("fallback.txt")
        return template.render()