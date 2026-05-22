"""
Entidades del dominio. Solo datos, sin lógica de infraestructura.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class PropertyType(str, Enum):
    CASA = "casa"
    APARTAMENTO = "apartamento"
    TERRENO = "terreno"
    LOCAL = "local"
    PENTHOUSE = "penthouse"
    # NUEVO Fase 1
    BIENHECHURIA = "bienhechuria"
    EDIFICIO = "edificio"
    OFICINA = "oficina"
    TOWNHOUSE = "townhouse"
    QUINTA = "quinta"
    LOCAL_COMERCIAL = "local_comercial"


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
    # NUEVO Fase 1: Columnas booleanas para filtros de búsqueda
    has_ocean_view: bool = False
    is_furnished: bool = False
    has_pool: bool = False
    has_parking: bool = False
    has_security: bool = False
    has_generator: bool = False
    has_water_tank: bool = False
    has_ac: bool = False
    is_new_construction: bool = False
    has_balcony: bool = False
    is_gated_community: bool = False
    # Campos de auditoría
    created_at: datetime | None = None
    updated_at: datetime | None = None


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
    has_rif: bool | None = None
    visit_planned: bool | None = None
    funding_source: str | None = None
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# NUEVO Fase 1: Cita de visita
class AppointmentStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class Appointment(BaseModel):
    """Cita de visita a propiedad."""
    id: int | None = None
    session_id: str
    property_id: int | None = None
    lead_id: int | None = None
    requested_date: str | None = None  # Ej: "mañana", "2026-05-25"
    requested_time: str | None = None  # Ej: "3pm", "15:00"
    status: AppointmentStatus = AppointmentStatus.PENDING
    notes: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None