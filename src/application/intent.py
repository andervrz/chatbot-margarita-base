"""
Detector de intenciones basado en regex + keywords.
Cero llamadas al LLM para clasificar. Cero dependencias externas.
"""

import re
from dataclasses import dataclass
from enum import Enum

from src.domain.models import Municipality, PropertyType


class IntentType(str, Enum):
    SEARCH_PROPERTY = "search_property"
    FAQ_PRICE_M2 = "faq_price_m2"
    FAQ_FOREIGN_BUY = "faq_foreign_buy"
    FAQ_RENTAL_ROI = "faq_rental_roi"
    FAQ_PROCEDURE = "faq_procedure"
    CAPTURE_LEAD = "capture_lead"
    BOOK_APPOINTMENT = "book_appointment"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


@dataclass
class ExtractedFilters:
    """Filtros extraídos del mensaje del usuario para búsqueda SQL."""
    municipality: Municipality | None = None
    zone: str | None = None
    property_type: PropertyType | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_bedrooms: int | None = None
    # NUEVO: Preferencias semánticas para Fase 1
    has_ocean_view: bool | None = None
    is_urban: bool | None = None
    is_budget_friendly: bool | None = None


@dataclass
class PartialLead:
    """Datos parciales de lead extraídos del mensaje."""
    name: str | None = None
    phone: str | None = None
    email: str | None = None


class IntentDetector:
    """
    Clasifica el mensaje del usuario y extrae filtros estructurados.
    Todo se resuelve con regex; no hay magia, no hay LLM involucrado.

    ORDEN DE PRECEDENCIA (crítico para no falsear búsquedas):
    1. Saludos y despedidas (muy específicos)
    2. FAQs (keywords temáticos; tienen prioridad sobre búsqueda incluso si hay zona)
    3. Captura de lead
    4. Agendamiento (interés concreto post-búsqueda)
    5. Búsqueda de propiedad (más general, va al final)
    """

    # ---------- Patrones de intención ----------

    _INTENT_PATTERNS: dict[IntentType, list[str]] = {
        # 1. Saludos y despedidas (muy específicos)
        IntentType.GREETING: [
            r"^hola[!\s]*",
            r"^buenos\s+días",
            r"^buenas\s+tardes",
            r"^buenas\s+noches",
            r"^saludos",
        ],
        IntentType.GOODBYE: [
            r"^gracias(\s+.*)?$",
            r"^adiós",
            r"^(muchas\s+)?gracias",
            r"^adiós\b",
            r"^hasta\s+luego",
            r"^nos\s+vemos",
            r"^chao",
        ],

        # 2. FAQs (patrones muy definidos; prioridad sobre SEARCH_PROPERTY)
        IntentType.FAQ_PRICE_M2: [
            r"precio\s+(por\s+)?m[2²]",
            r"precio\s+(por\s+)?metro\s+cuadrado",
            r"cuánto\s+cuesta\s+(el\s+)?m[2²]",
            r"cuánto\s+cuesta\s+(el\s+)?metro\s+cuadrado",
            r"cuanto\s+cuesta\s+(el\s+)?m[2²]",
            r"cuanto\s+cuesta\s+(el\s+)?metro\s+cuadrado",
            r"valor\s+(del\s+)?metro\s+cuadrado",
            r"costo\s+por\s+m[2²]",
        ],
        IntentType.FAQ_FOREIGN_BUY: [
            r"extranjero\s+(puede\s+)?comprar",
            r"soy\s+extranjero",
            r"no\s+soy\s+venezolano",
            r"visado\s+de\s+inversionista",
            r"visa\s+de\s+inversionista",
            r"pasaporte\s+extranjero",
            r"comprar\s+desde\s+(el\s+)?exterior",
            r"comprar\s+sin\s+estar\s+en\s+venezuela",
            r"comprar\s+desde\s+(usa|españa|miami|madrid|estados\s+unidos)",
        ],
        IntentType.FAQ_RENTAL_ROI: [
            r"rentabilidad",
            r"retorno\s+de\s+inversión",
            r"retorno\s+(de\s+)?inversión",
            r"roi",
            r"alquiler\s+vacacional\s+rentable",
            r"cuánto\s+(se\s+)?(gana|renta|produce)",
            r"airbnb",
        ],
        IntentType.FAQ_PROCEDURE: [
            r"trámites?\s+(de\s+)?compra",
            r"proceso\s+(de\s+)?compra",
            r"documentos?\s+necesarios?",
            r"pasos?\s+para\s+comprar",
            r"escritura",
            r"registro\s+de\s+la\s+propiedad",
        ],

        # 3. Captura de lead
        IntentType.CAPTURE_LEAD: [
            r"mi\s+nombre\s+es\s+([a-záéíóúñ\s]+)",
            r"me\s+llamo\s+([a-záéíóúñ\s]+)",
            r"me\s+llamo\s+([a-záéíóúñ\s]+?)(?:\s+y\s+|$)",
            r"mi\s+(teléfono|celular|correo|email|whatsapp|contacto)\s+es",
            r"contactenme",
            r"quiero\s+que\s+me\s+llamen",
            r"quiero\s+que\s+(un\s+asesor\s+)?me\s+llamen?",
            r"mi\s+(cel|tlf|wp)\s+es",
        ],

        # 4. Agendamiento (NUEVO Fase 1)
        IntentType.BOOK_APPOINTMENT: [
            r"me\s+interesa",
            r"quiero\s+verla",
            r"quiero\s+ver",
            r"agendar",
            r"programar\s+visita",
            r"cuándo\s+puedo\s+ver",
            r"cuando\s+puedo\s+ver",
            r"visitar\s+(la\s+)?propiedad",
            r"ir\s+a\s+ver",
            r"coordinar\s+visita",
        ],

        # 5. Búsqueda de propiedad (más general, va al final)
        IntentType.SEARCH_PROPERTY: [
            r"(busco|quiero|necesito|estoy\s+buscando|me\s+interesa)\s+(?:una\s+)?(casa|apartamento|apto|terreno|local|penthouse|bienhechuría|edificio|oficina|townhouse|quinta|local\s+comercial)",
            r"(busco|quiero|necesito|estoy\s+buscando|me\s+interesa)\s+(?:algo\s+)?en\s+\w+",
            r"(dónde\s+hay|hay\s+algún|hay\s+alguna)\s+(casa|apartamento|apto|terreno|local|penthouse|bienhechuría|edificio|oficina|townhouse|quinta)",
            r"(apartamento|casa|terreno|local|penthouse|bienhechuría|edificio|oficina|townhouse|quinta)\s+(?:en\s+)?\w+",
            r"(busco|quiero|necesito)\s+(?:una\s+)?(propiedad|inmueble|vivienda)",
        ],
    }

    # ---------- Mapa de zona → municipio ----------
    _DEFAULT_ZONE_MAP: dict[str, tuple[Municipality, str]] = {
        # Maneiro
        "altos de maneiro": (Municipality.MANEIRO, "Altos de Maneiro"),
        "playa el ángel": (Municipality.MANEIRO, "Playa El Ángel"),
        "pampatar": (Municipality.MANEIRO, "Pampatar"),
        # Mariño
        "porlamar centro": (Municipality.MARINO, "Porlamar Centro"),
        "costa azul": (Municipality.MARINO, "Costa Azul"),
        "playa moreno": (Municipality.MARINO, "Playa Moreno"),
        "sabanamar": (Municipality.MARINO, "Sabanamar"),
        "san lorenzo": (Municipality.MARINO, "San Lorenzo"),
        "guatamare": (Municipality.MARINO, "Guatamare"),
        "porlamar": (Municipality.MARINO, "Porlamar"),
        # Antolín del Campo
        "playa el agua": (Municipality.ANTOLIN_DEL_CAMPO, "Playa El Agua"),
        "la mira": (Municipality.ANTOLIN_DEL_CAMPO, "La Mira"),
        # Arismendi
        "la asunción": (Municipality.ARISMENDI, "La Asunción"),
        "juan griego": (Municipality.ARISMENDI, "Juan Griego"),
        "el tirano": (Municipality.ARISMENDI, "El Tirano"),
        "la guardia": (Municipality.ARISMENDI, "La Guardia"),
        # Gómez
        "playa caribe": (Municipality.GOMEZ, "Playa Caribe"),
        "guacuco": (Municipality.GOMEZ, "Guacuco"),
        # Marcano
        "playa parguito": (Municipality.MARCANO, "Playa Parguito"),
        "la caranta": (Municipality.MARCANO, "La Caranta"),
        # Municipios sin zona específica (fallbacks)
        "maneiro": (Municipality.MANEIRO, "Pampatar"),
        "mariño": (Municipality.MARINO, "Porlamar"),
        "arismendi": (Municipality.ARISMENDI, "La Asunción"),
        "antolín del campo": (Municipality.ANTOLIN_DEL_CAMPO, "Playa El Agua"),
        "antolin del campo": (Municipality.ANTOLIN_DEL_CAMPO, "Playa El Agua"),
        "gómez": (Municipality.GOMEZ, "Playa Caribe"),
        "gomez": (Municipality.GOMEZ, "Playa Caribe"),
        "marcano": (Municipality.MARCANO, "Playa Parguito"),
        # Zonas adicionales del mercado real
        "apostadero": (Municipality.MANEIRO, "Apostadero"),
        "jorge coll": (Municipality.MARINO, "Jorge Coll"),
        "paraíso": (Municipality.MARINO, "Paraíso"),
        "los geranios": (Municipality.MARINO, "Los Geranios"),
        "los peregrinos": (Municipality.MARINO, "Los Peregrinos"),
        "vincenzo": (Municipality.MARINO, "Vincenzo"),
        "lomas de encanto": (Municipality.MARINO, "Lomas de Encanto"),
        "el horcón": (Municipality.MARINO, "El Horcón"),
        "macanao": (Municipality.PENINSULA_DE_MACANAO, "Macanao"),
        "robledal": (Municipality.PENINSULA_DE_MACANAO, "Robledal"),
        "playa el yaque": (Municipality.PENINSULA_DE_MACANAO, "Playa El Yaque"),
    }

    # ---------- Respuestas predefinidas ----------

    FAQ_RESPONSES: dict[IntentType, str] = {
        IntentType.FAQ_PRICE_M2: (
            "Los precios por m² en Isla de Margarita varían según la zona:\n\n"
            "• **Maneiro** (Pampatar, Playa El Ángel): $1,000 – $1,500/m²\n"
            "• **Mariño** (Porlamar): $600 – $1,000/m²\n"
            "• **Antolín del Campo** (Playa El Agua): $700 – $1,200/m²\n"
            "• **Arismendi** (La Asunción): $400 – $800/m²\n"
            "• **Gómez** (Playa Caribe): $300 – $600/m²\n\n"
            "Estos son rangos referenciales. ¿Te gustaría que te envíe opciones específicas en alguna zona?"
        ),
        IntentType.FAQ_FOREIGN_BUY: (
            "Sí, los extranjeros pueden comprar propiedades en Margarita. "
            "Se requiere principalmente:\n\n"
            "• **Visa de transeúnte** (negocios, rentista o familiar)\n"
            "• **RIF venezolano** (se tramita con pasaporte)\n"
            "• **ZODI-71** (trámite personal en la Isla, no delegable)\n"
            "• **Asesoría legal local** para la escritura y registro\n\n"
            "Los costos adicionales suelen ser del 5-10% sobre el precio de venta. "
            "¿Te gustaría que un asesor especializado te contacte para guiarte paso a paso?"
        ),
        IntentType.FAQ_RENTAL_ROI: (
            "El alquiler vacacional en Margarita puede ser muy rentable, especialmente en:\n\n"
            "• **Playa El Agua / Pampatar**: ocupación alta todo el año\n"
            "• **Porlamar**: demanda de ejecutivos y turistas\n\n"
            "El ROI anual estimado oscila entre el **6% y 12%** según la zona y gestión. "
            "¿Buscas una propiedad específica para inversión?"
        ),
        IntentType.FAQ_PROCEDURE: (
            "El proceso de compra en Margarita generalmente incluye:\n\n"
            "1. **Negociación y reserva** (10-20% del precio)\n"
            "2. **Revisión de documentos** (solvencia, certificado de tradición)\n"
            "3. **Firma de escritura pública** ante notario\n"
            "4. **Registro en la Oficina Subalterna** correspondiente\n"
            "5. **Pago de impuestos** (timbres fiscales, registro)\n\n"
            "Tiempo estimado: 30-60 días. ¿Necesitas ayuda con algún paso específico?"
        ),
        IntentType.GREETING: (
            "¡Hola! Soy tu asesor inmobiliario de Margarita. "
            "¿En qué puedo ayudarte hoy? Puedo buscarte propiedades, "
            "informarte precios por zona o ayudarte con trámites de compra."
        ),
        IntentType.GOODBYE: (
            "¡Gracias por contactarnos! Estoy aquí cuando lo necesites. "
            "Si quieres retomar esta conversación más tarde, solo escríbeme. ¡Que tengas un excelente día!"
        ),
    }

    # ---------- Métodos públicos ----------

    def __init__(self, zone_map: dict[str, tuple[Municipality, str]] | None = None) -> None:
        self._zone_map = zone_map or self._DEFAULT_ZONE_MAP

    def detect(self, message: str) -> IntentType:
        """Clasifica la intención del mensaje."""
        text_lower = message.lower().strip()
        for intent, patterns in self._INTENT_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                return intent
        return IntentType.UNKNOWN

    def get_faq_response(self, intent: IntentType) -> str | None:
        """Retorna respuesta predefinida si la intención es una FAQ o saludo."""
        return self.FAQ_RESPONSES.get(intent)

    def extract_lead_info(self, message: str) -> PartialLead:
        """Extrae datos de contacto del mensaje para captura de lead."""
        text_lower = message.lower()
        lead = PartialLead()

        # Nombre
        name_match = re.search(
            r"mi\s+nombre\s+es\s+([a-záéíóúñ\s]+?)(?:$|\.|,|mi\s|y\s|tel|correo|contacto|whatsapp|wp)",
            text_lower,
        )
        if not name_match:
            name_match = re.search(
                r"me\s+llamo\s+([a-záéíóúñ\s]+?)(?:$|\.|,|mi\s|y\s|tel|correo|contacto|whatsapp|wp)",
                text_lower,
            )
        if name_match:
            lead.name = name_match.group(1).strip().title()

        # Teléfono - normalizar espacios y guiones primero
        phone_text = re.sub(r"(?<=\d)\s+(?=\d)", "", message)
        phone_text_lower = phone_text.lower()

        phone_match = re.search(
            r"(?:teléfono|celular|tlf|phone|cel|contacto|whatsapp|wp)\s*(?:es|:|al)?\s*([\d\-+()]{7,})",
            phone_text_lower,
        )
        if phone_match:
            raw = phone_match.group(1)
            digits = re.sub(r"[^\d]", "", raw)
            lead.phone = self._normalize_phone(digits)
        else:
            standalone = re.search(
                r"(?:\+?58)?\s*(?:0)?(4(?:12|14|16|24|26)[\d\-]{7,})",
                phone_text,
            )
            if standalone:
                raw = standalone.group(0)
                digits = re.sub(r"[^\d]", "", raw)
                lead.phone = self._normalize_phone(digits)

        # Email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w{2,}", message)
        if email_match:
            lead.email = email_match.group(0).lower()

        return lead

    def extract_filters(self, message: str) -> ExtractedFilters:
        """Extrae filtros estructurados del mensaje del usuario."""
        text_lower = message.lower()
        filters = ExtractedFilters()

        # Zona y Municipio
        sorted_zones = sorted(self._zone_map.items(), key=lambda x: -len(x[0]))
        for key, (municipality, zone) in sorted_zones:
            if key in text_lower:
                filters.municipality = municipality
                filters.zone = zone
                break

        # Tipo de propiedad (NUEVOS TIPOS Fase 1)
        type_map = {
            "casa": PropertyType.CASA,
            "apartamento": PropertyType.APARTAMENTO,
            "apto": PropertyType.APARTAMENTO,
            "terreno": PropertyType.TERRENO,
            "local": PropertyType.LOCAL,
            "local comercial": PropertyType.LOCAL_COMERCIAL,
            "penthouse": PropertyType.PENTHOUSE,
            "bienhechuría": PropertyType.BIENHECHURIA,
            "bienhechuria": PropertyType.BIENHECHURIA,
            "edificio": PropertyType.EDIFICIO,
            "oficina": PropertyType.OFICINA,
            "townhouse": PropertyType.TOWNHOUSE,
            "quinta": PropertyType.QUINTA,
        }
        # Ordenar por longitud descendente para que "local comercial" gane sobre "local"
        for key, value in sorted(type_map.items(), key=lambda x: -len(x[0])):
            if key in text_lower:
                filters.property_type = value
                break

        # Precio
        price_patterns = [
            r".*?\b(?:entre|de)\s+[\$]?\s*(\d[\d\.,]*(?:\s*(?:k|mil))?)\s+y\s+[\$]?\s*(\d[\d\.,]*(?:\s*(?:k|mil))?)\s*(?:usd|dólares|dolares)?",
            r"(?:hasta|menos\s+de|máximo|maximo)\s+[\$]?\s*(\d[\d\.,]*(?:\s*(?:k|mil))?)\s*(?:usd|dólares|dolares)?",
            r"(?:precio|presupuesto|valor)\s+(?:de\s+)?[\$]?\s*(\d[\d\.,]*(?:\s*(?:k|mil))?)\s*(?:usd|dólares|dolares)?",
            r"[\$]?\s*(\d[\d\.,]*(?:\s*(?:k|mil))?)\s*(?:usd|dólares|dolares)?",
        ]
        for pattern in price_patterns:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    filters.min_price = self._parse_price(groups[0], text_lower)
                    filters.max_price = self._parse_price(groups[1], text_lower)
                else:
                    filters.max_price = self._parse_price(groups[0], text_lower)
                break

        # Habitaciones
        bed_match = re.search(r"(\d+)\s*(?:habitaciones|hab|cuartos|recámaras)", text_lower)
        if bed_match:
            filters.min_bedrooms = int(bed_match.group(1))

        # NUEVO Fase 1: Preferencias semánticas
        if any(term in text_lower for term in ["vista al mar", "con vista al mar", "frente al mar", "vista al océano", "ocean view"]):
            filters.has_ocean_view = True
        if any(term in text_lower for term in ["zona céntrica", "céntrico", "urbano", "centro", "centrico"]):
            filters.is_urban = True
        if any(term in text_lower for term in ["económico", "economico", "barato", "accesible", "bajo precio", "precio bajo", "barata"]):
            filters.is_budget_friendly = True

        return filters

    # ---------- Helpers privados ----------

    @staticmethod
    def _parse_price(text: str, full_context: str = "") -> float | None:
        """Limpia y convierte un string de precio a float."""
        try:
            original = text.lower().strip()
            if not original:
                return None

            context = (original + " " + full_context).lower()
            has_k_or_mil = "k" in context or "mil" in context

            clean = original
            has_k_suffix = False

            if re.search(r"\d\s*k\b", clean):
                has_k_suffix = True
                clean = re.sub(r"(\d)\s*k\b", r"\1", clean)
            elif re.search(r"\d\s*mil\b", clean):
                has_k_suffix = True
                clean = re.sub(r"(\d)\s*mil\b", r"\1", clean)

            clean = clean.replace("$", "").replace("usd", "").replace("dólares", "").replace("dolares", "")

            if clean.count(".") > 1:
                clean = clean.replace(".", "")
            elif clean.count(".") == 1:
                parts = clean.split(".")
                if len(parts) == 2 and len(parts[1]) == 3 and parts[1].isdigit():
                    clean = clean.replace(".", "")

            clean = clean.replace(",", "").strip()
            val = float(clean)

            if has_k_suffix:
                val *= 1000
            elif val < 1000 and has_k_or_mil and val >= 10:
                val *= 1000
            elif val < 100:
                val *= 1000

            return val
        except (ValueError, AttributeError):
            return None

    def _normalize_phone(self, digits: str) -> str | None:
        """Normaliza un número telefónico venezolano."""
        if not digits or len(digits) < 10:
            return None

        if digits.startswith("58") and len(digits) >= 12:
            return digits

        if digits.startswith("0") and len(digits) == 11 and digits[1] == "4":
            return digits

        if len(digits) == 10 and digits.startswith("4"):
            return "0" + digits

        if len(digits) == 11:
            return digits

        return digits