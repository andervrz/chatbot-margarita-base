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
    4. Búsqueda de propiedad (más general, va al final)
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
            r"^hasta\s+luego",
            r"^nos\s+vemos",
            r"^chao",
        ],

        # 2. FAQs (patrones muy definidos; prioridad sobre SEARCH_PROPERTY)
        # Incluimos variantes ASCII (m2) y Unicode (m²) para precios
        IntentType.FAQ_PRICE_M2: [
            r"precio\s+(por\s+)?m[2²]",
            r"precio\s+(por\s+)?metro\s+cuadrado",
            r"cuánto\s+cuesta\s+(el\s+)?m[2²]",
            r"cuánto\s+cuesta\s+(el\s+)?metro\s+cuadrado",
            r"valor\s+(del\s+)?metro\s+cuadrado",
            r"costo\s+por\s+m[2²]",
        ],
        IntentType.FAQ_FOREIGN_BUY: [
            r"extranjero\s+(puede\s+)?comprar",
            r"no\s+soy\s+venezolano",
            r"visado\s+de\s+inversionista",
            r"pasaporte\s+extranjero",
            r"comprar\s+desde\s+(el\s+)?exterior",
        ],
        IntentType.FAQ_RENTAL_ROI: [
            r"rentabilidad",
            r"retorno\s+de\s+inversión",
            r"roi",
            r"alquiler\s+vacacional\s+rentable",
            r"cuánto\s+(se\s+)?(gana|renta|produce)",
        ],
        IntentType.FAQ_PROCEDURE: [
            r"trámites?\s+(de\s+)?compra",
            r"documentos?\s+necesarios?",
            r"pasos?\s+para\s+comprar",
            r"escritura",
            r"registro\s+de\s+la\s+propiedad",
        ],

        # 3. Captura de lead
        IntentType.CAPTURE_LEAD: [
            r"mi\s+nombre\s+es\s+([a-záéíóúñ\s]+)",
            r"me\s+llamo\s+([a-záéíóúñ\s]+)",
            r"mi\s+(teléfono|celular|correo|email)\s+es",
            r"contactenme",
            r"quiero\s+que\s+me\s+llamen",
        ],

        # 4. Búsqueda de propiedad (más general, va al final)
        # Requiere presencia de VERBO de búsqueda + sustantivo de propiedad,
        # o verbo de búsqueda + preposición de ubicación.
        IntentType.SEARCH_PROPERTY: [
            r"(busco|quiero|necesito|estoy buscando|me interesa)\s+(?:una\s+)?(casa|apartamento|apto|terreno|local|penthouse)",
            r"(busco|quiero|necesito|estoy buscando|me interesa)\s+(?:algo\s+)?en\s+\w+",
            r"(dónde\s+hay|hay\s+algún|hay\s+alguna)\s+(casa|apartamento|apto|terreno|local|penthouse)",
        ],
    }

    # ---------- Mapa de zona → municipio (fuente única de verdad) ----------
    # Cada zona mapea a (municipality_enum, zone_display_name)
    _ZONE_MAP: dict[str, tuple[Municipality, str]] = {
        # Maneiro
        "pampatar": (Municipality.MANEIRO, "Pampatar"),
        "playa el ángel": (Municipality.MANEIRO, "Playa El Ángel"),
        # Mariño
        "porlamar": (Municipality.MARINO, "Porlamar"),
        "porlamar centro": (Municipality.MARINO, "Porlamar Centro"),
        "sabanamar": (Municipality.MARINO, "Sabanamar"),
        "costa azul": (Municipality.MARINO, "Costa Azul"),
        "playa moreno": (Municipality.MARINO, "Playa Moreno"),
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
        # Municipios sin zona específica en seed
        "maneiro": (Municipality.MANEIRO, "Pampatar"),
        "mariño": (Municipality.MARINO, "Porlamar"),
        "arismendi": (Municipality.ARISMENDI, "La Asunción"),
        "antolín del campo": (Municipality.ANTOLIN_DEL_CAMPO, "Playa El Agua"),
        "antolin del campo": (Municipality.ANTOLIN_DEL_CAMPO, "Playa El Agua"),
        "gómez": (Municipality.GOMEZ, "Playa Caribe"),
        "gomez": (Municipality.GOMEZ, "Playa Caribe"),
        "marcano": (Municipality.MARCANO, "Playa Parguito"),
    }

    # ---------- Respuestas predefinidas (0 LLM calls) ----------

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
            "• **Visa de inversionista** o residencia temporal\n"
            "• **RIF venezolano** (se tramita con pasaporte)\n"
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
            "5. **Pago de impuestos** (ISLR, registro)\n\n"
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
        name_match = re.search(r"mi\s+nombre\s+es\s+([a-záéíóúñ\s]+?)(?:$|\.|,|mi\s|y\s|tel|correo)", text_lower)
        if not name_match:
            name_match = re.search(r"me\s+llamo\s+([a-záéíóúñ\s]+?)(?:$|\.|,|mi\s|y\s|tel|correo)", text_lower)
        if name_match:
            lead.name = name_match.group(1).strip().title()

        # Teléfono
        phone_match = re.search(r"(?:teléfono|celular|tlf|phone)\s*(?:es|:)?\s*([\d\s\-+()]{7,})", text_lower)
        if phone_match:
            lead.phone = re.sub(r"[^\d]", "", phone_match.group(1))[:15]

        # Email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w{2,}", message)
        if email_match:
            lead.email = email_match.group(0).lower()

        return lead

    def extract_filters(self, message: str) -> ExtractedFilters:
        """
        Extrae filtros estructurados del mensaje del usuario.
        Esto permite hacer búsquedas SQL exactas sin depender del LLM.
        """
        text_lower = message.lower()
        filters = ExtractedFilters()

        # --- Zona y Municipio (fuente única: _ZONE_MAP) ---
        for key, (municipality, zone) in self._ZONE_MAP.items():
            if key in text_lower:
                filters.municipality = municipality
                filters.zone = zone
                break

        # --- Tipo de propiedad ---
        type_map = {
            "casa": PropertyType.CASA,
            "apartamento": PropertyType.APARTAMENTO,
            "apto": PropertyType.APARTAMENTO,
            "terreno": PropertyType.TERRENO,
            "local": PropertyType.LOCAL,
            "penthouse": PropertyType.PENTHOUSE,
        }
        for key, value in type_map.items():
            if key in text_lower:
                filters.property_type = value
                break

        # --- Precio (regex robusto con contexto completo para _parse_price) ---
        price_patterns = [
            # Rango: "entre X y Y", "de X a Y"
            r"(?:entre|de)\s+[\$]?\s*(\d[\d\.,]*(?:\s*(?:k|mil))?)\s+y\s+[\$]?\s*(\d[\d\.,]*(?:\s*(?:k|mil))?)\s*(?:usd|dólares|dolares)?",
            # Hasta/máximo
            r"(?:hasta|menos\s+de|máximo|maximo)\s+[\$]?\s*(\d[\d\.,]*(?:\s*(?:k|mil))?)\s*(?:usd|dólares|dolares)?",
            # Precio suelto (cuando hay contexto de búsqueda implícito)
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
                    # Heurística: si hay "entre/de" o "hasta", interpretar como max
                    # Si solo hay número suelto con verbo de búsqueda, también max (presupuesto)
                    filters.max_price = self._parse_price(groups[0], text_lower)
                break

        # --- Habitaciones ---
        bed_match = re.search(r"(\d+)\s*(?:habitaciones|hab|cuartos|recámaras)", text_lower)
        if bed_match:
            filters.min_bedrooms = int(bed_match.group(1))

        return filters

    # ---------- Helpers privados ----------

    @staticmethod
    def _parse_price(text: str, full_context: str = "") -> float | None:
        """Limpia y convierte un string de precio a float.

        Soporta:
        - Separadores de miles: 95.000, 95,000
        - Sufijo 'k' o 'mil' pegados o separados: 70k → 70000, 60 mil → 60000, 40 k → 40000
        - Moneda: $, usd, dólares
        - Contexto inmobiliario: números < 1000 en contexto con 'mil'/'k' se multiplican por 1000

        Args:
            text: El grupo capturado por regex (puede no incluir 'mil'/'k')
            full_context: El mensaje completo para detectar contexto de miles

        Limitación: asume que no hay centavos (precios inmobiliarios en Margarita).
        """
        try:
            original = text.lower().strip()
            clean = original.replace("$", "").replace("usd", "").replace("dólares", "").replace("dolares", "")

            # Detectar si hay 'k' o 'mil' en el texto original o en el contexto completo
            context = (original + " " + full_context).lower()
            has_k_or_mil = "k" in context or "mil" in context

            # Normalizar 'k' y 'mil' (pegados o separados por espacio)
            clean = re.sub(r'(\d)\s*k\b', r'\1000', clean)
            clean = re.sub(r'(\d)\s*mil\b', r'\1000', clean)

            # Eliminar separadores de miles (punto o coma)
            clean = clean.replace(".", "").replace(",", "").strip()

            val = float(clean)

            # Heurística inmobiliaria: en Margarita no hay propiedades < $1,000.
            # Si el valor es < 1000 y el contexto sugiere miles (por 'k'/'mil' o por ser 
            # número redondo típico de presupuesto), multiplicar por 1000.
            if val < 1000 and has_k_or_mil:
                val *= 1000
            elif val < 100:
                # Números < 100 en contexto inmobiliario son siempre miles
                val *= 1000

            return val
        except (ValueError, AttributeError):
            return None
