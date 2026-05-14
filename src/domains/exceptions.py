"""Excepciones del dominio."""


class DomainError(Exception):
    """Base para todas las excepciones del dominio."""
    pass


class PropertyNotFound(DomainError):
    """No se encontró la propiedad solicitada."""
    pass


class NoDataError(DomainError):
    """No hay datos disponibles para responder."""
    pass


class LLMError(DomainError):
    """Error en la comunicación con el LLM."""
    pass
