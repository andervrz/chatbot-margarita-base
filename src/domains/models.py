"""
Entidades del dominio. Solo datos, sin lógica de infraestructura.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PropertyType(str, Enum):
    CASA = "casa"
    APARTAMENTO = "apartamento"
    TERRENO = "terreno"
    LOCAL = "local"
    PENTHOUSE = "penthouse"


class PropertyStatus(str, Enum):
    AVAILABLE = "available"
    SOLD = "sold"
    RENTED = "rented"
    RESERVED = "reserved"


class Municipality(str, Enum):
    MANEIRO = "Maneiro"
    MARINO = "Mariño"
    ARISMENDI = "Arismendi"
    ANTOLIN_DEL_CAMPO = "Antolín del Campo"
    GOMEZ = "Gómez"
    DIAZ = "Díaz"
    GARCIA = "García"
    MARCANO = "Marcano"
    PENINSULA_DE_MACANAO = "Península de Macanao"


class Property(BaseModel):
    """Propiedad inmobiliaria. Fuente de verdad en SQLite."""
    id: int | None = None
    title: str
    municipality: Municipality
    zone: str
    type: PropertyType
    price_usd: float | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    area_m2: float | None = None
    description: str = ""
    features: list[str] = Field(default_factory=list)
    status: PropertyStatus = PropertyStatus.AVAILABLE
    contact_phone: str | None = None
    contact_email: str | None = None


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Mensaje de conversación."""
    role: MessageRole
    content: str
    metadata: dict = Field(default_factory=dict)


class ConversationSession(BaseModel):
    """Sesión de conversación con historial."""
    session_id: str
    messages: list[Message] = Field(default_factory=list)
    user_id: str | None = None


class LeadInterest(str, Enum):
    COMPRA = "compra"
    ALQUILER = "alquiler"
    INVERSION = "inversion"
    ALQUILER_VACACIONAL = "alquiler_vacacional"


class LeadUrgency(str, Enum):
    INMEDIATA = "inmediata"
    TRES_MESES = "3_meses"
    SEIS_MESES = "6_meses"
    SOLO_MIRANDO = "solo_mirando"


class Lead(BaseModel):
    """Lead capturado durante la conversación."""
    id: int | None = None
    session_id: str
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    interest_type: LeadInterest | None = None
    budget_usd: float | None = None
    preferred_zone: str | None = None
    preferred_type: PropertyType | None = None
    urgency: LeadUrgency | None = None
    captured_at: datetime = Field(default_factory=datetime.utcnow)
