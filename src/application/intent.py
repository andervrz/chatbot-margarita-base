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


class IntentDetector:
    """
    Clasifica el mensaje del usuario y extrae filtros estructurados.
    Todo se resuelve con regex; no hay magia, no hay LLM involucrado.
    """

    # ---------- Patrones de intención ----------

    _INTENT_PATTERNS: dict[IntentType, list[str]] = {
        IntentType.SEARCH_PROPERTY: [
            r"(busco|quiero|necesito|estoy buscando|me interesa|dónde hay|hay algún|hay alguna)",
            r"(casa|apartamento|terreno|local|penthouse|anexo|townhouse)",
            r"(en\s+\w+|cerca de\s+\w+|zona\s+\w+)",
        ],
        IntentType.FAQ_PRICE_M2: [
            r"precio\s+(por\s+)?m²",
            r"precio\s+(por\s+)?metro\s+cuadrado",
            r"cuánto\s+cuesta\s+(el\s+)?m²",
            r"valor\s+(del\s+)?metro\s+cuadrado",
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
        IntentType.CAPTURE_LEAD: [
            r"mi\s+nombre\s+es\s+([a-záéíóúñ\s]+)",
            r"me\s+llamo\s+([a-záéíóúñ\s]+)",
            r"mi\s+(teléfono|celular|correo|email)\s+es",
            r"contactenme",
            r"quiero\s+que\s+me\s+llamen",
        ],
        IntentType.GREETING: [
            r"^hola[!\s]*",
            r"^buenos\s+días",
            r"^buenas\s+tardes",
            r"^buenas\s+noches",
            r"^saludos",
        ],
        IntentType.GOODBYE: [
            r"gracias[!\s]*$",
            r"^adiós",
            r"^hasta\s+luego",
            r"^nos\s+vemos",
            r"^chao",
        ],
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

    def extract_filters(self, message: str) -> ExtractedFilters:
        """
        Extrae filtros estructurados del mensaje del usuario.
        Esto permite hacer búsquedas SQL exactas sin depender del LLM.
        """
        text_lower = message.lower()
        filters = ExtractedFilters()

        # --- Municipio ---
        municipality_map = {
            "maneiro": Municipality.MANEIRO,
            "pampatar": Municipality.MANEIRO,
            "playa el ángel": Municipality.MANEIRO,
            "mariño": Municipality.MARINO,
            "porlamar": Municipality.MARINO,
            "arismendi": Municipality.ARISMENDI,
            "la asunción": Municipality.ARISMENDI,
            "antolín del campo": Municipality.ANTOLIN_DEL_CAMPO,
            "antolin del campo": Municipality.ANTOLIN_DEL_CAMPO,
            "playa el agua": Municipality.ANTOLIN_DEL_CAMPO,
            "gómez": Municipality.GOMEZ,
            "gomez": Municipality.GOMEZ,
            "playa caribe": Municipality.GOMEZ,
        }
        for key, value in municipality_map.items():
            if key in text_lower:
                filters.municipality = value
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

        # --- Precio (regex simple) ---
        # "hasta 50000", "menos de 80000", "entre 30000 y 60000"
        price_patterns = [
            r"(?:hasta|menos\s+de|máximo|maximo)\s+[\$]?\s*(\d[\d\.,]*)",
            r"(?:entre|de)\s+[\$]?\s*(\d[\d\.,]*)\s+y\s+[\$]?\s*(\d[\d\.,]*)",
            r"[\$]?\s*(\d[\d\.,]*)\s*(?:usd|dólares|dolares)?",
        ]
        for pattern in price_patterns:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    filters.min_price = self._parse_price(groups[0])
                    filters.max_price = self._parse_price(groups[1])
                else:
                    filters.max_price = self._parse_price(groups[0])
                break

        # --- Habitaciones ---
        bed_match = re.search(r"(\d+)\s*(?:habitaciones|hab|cuartos|recámaras)", text_lower)
        if bed_match:
            filters.min_bedrooms = int(bed_match.group(1))

        return filters

    # ---------- Helpers privados ----------

    @staticmethod
    def _parse_price(text: str) -> float | None:
        """Limpia y convierte un string de precio a float."""
        try:
            clean = text.replace(".", "").replace(",", "").replace("$", "").strip()
            return float(clean)
        except (ValueError, AttributeError):
            return None
