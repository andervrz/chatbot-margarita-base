"""
Enriquecimiento de prompts.
Toma datos reales de la base de datos y los inyecta en templates Jinja2.
El LLM solo formatea; la información viene estrictamente de SQLite.
"""

from jinja2 import Environment, PackageLoader, select_autoescape

from src.domain.models import Property


class PromptEnricher:
    """
    Construye prompts listos para el LLM usando templates Jinja2.
    Separar los prompts del código Python es clave para mantenibilidad.
    """

    def __init__(self) -> None:
        self.env = Environment(
            loader=PackageLoader("src", "prompts"),
            autoescape=select_autoescape(),
        )

    def build_search_context(
        self,
        properties: list[Property],
        user_query: str,
    ) -> str:
        """
        Construye el contexto de búsqueda para el LLM.
        Cada propiedad incluye datos verificados de la base de datos.
        """
        template = self.env.get_template("search_results.txt.j2")
        return template.render(
            user_query=user_query,
            properties=properties,
            count=len(properties),
        )

    def build_vector_context(
        self,
        properties: list[Property],
        user_query: str,
    ) -> str:
        """
        Contexto cuando la búsqueda fue vectorial (fallback).
        Indica al LLM que los resultados son aproximados por similitud semántica.
        """
        template = self.env.get_template("vector_results.txt.j2")
        return template.render(
            user_query=user_query,
            properties=properties,
            count=len(properties),
        )

    def build_system_prompt(self) -> str:
        """Carga el system prompt base desde archivo externo."""
        template = self.env.get_template("system_v1.txt")
        return template.render()

    def build_fallback_message(self) -> str:
        """Mensaje controlado cuando no hay datos en ningún lado."""
        template = self.env.get_template("fallback.txt")
        return template.render()
