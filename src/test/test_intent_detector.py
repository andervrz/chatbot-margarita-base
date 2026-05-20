"""
Tests del detector de intenciones.
Verifica clasificación regex y extracción de filtros estructurados.
"""

import pytest

from src.application.intent import ExtractedFilters, IntentDetector, IntentType
from src.domain.models import Municipality, PropertyType


class TestIntentDetection:
    def setup_method(self):
        self.detector = IntentDetector()

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("hola, buenos días", IntentType.GREETING),
            ("HOLA!", IntentType.GREETING),
            ("busco una casa en pampatar", IntentType.SEARCH_PROPERTY),
            ("quiero apartamento en porlamar", IntentType.SEARCH_PROPERTY),
            ("precio por metro cuadrado", IntentType.FAQ_PRICE_M2),
            ("cuánto cuesta el m² en playa el agua", IntentType.FAQ_PRICE_M2),
            ("extranjero puede comprar", IntentType.FAQ_FOREIGN_BUY),
            ("no soy venezolano, puedo comprar", IntentType.FAQ_FOREIGN_BUY),
            ("gracias por la info", IntentType.GOODBYE),
            ("adiós", IntentType.GOODBYE),
            ("mi nombre es Carlos", IntentType.CAPTURE_LEAD),
            ("rentabilidad de alquiler vacacional", IntentType.FAQ_RENTAL_ROI),
            ("documentos necesarios para comprar", IntentType.FAQ_PROCEDURE),
        ],
    )
    def test_detect(self, message: str, expected: IntentType):
        assert self.detector.detect(message) == expected

    def test_faq_response_returns_text(self):
        response = self.detector.get_faq_response(IntentType.FAQ_PRICE_M2)
        assert response is not None
        assert "$1,000" in response

    def test_faq_response_none_for_unknown(self):
        assert self.detector.get_faq_response(IntentType.UNKNOWN) is None


class TestFilterExtraction:
    def setup_method(self):
        self.detector = IntentDetector()

    def test_extract_municipality_pampatar(self):
        filters = self.detector.extract_filters("casa en pampatar")
        assert filters.municipality == Municipality.MANEIRO

    def test_extract_municipality_porlamar(self):
        filters = self.detector.extract_filters("apartamento porlamar")
        assert filters.municipality == Municipality.MARINO

    def test_extract_property_type(self):
        filters = self.detector.extract_filters("busco terreno")
        assert filters.property_type == PropertyType.TERRENO

    def test_extract_price_range(self):
        filters = self.detector.extract_filters("hasta 80000 dolares")
        assert filters.max_price == 80000

    def test_extract_bedrooms(self):
        filters = self.detector.extract_filters("3 habitaciones")
        assert filters.min_bedrooms == 3

    def test_extract_multiple_filters(self):
        filters = self.detector.extract_filters(
            "apartamento en playa el agua hasta 100000 con 2 habitaciones"
        )
        assert filters.property_type == PropertyType.APARTAMENTO
        assert filters.municipality == Municipality.ANTOLIN_DEL_CAMPO
        assert filters.max_price == 100000
        assert filters.min_bedrooms == 2

    def test_no_filters(self):
        filters = self.detector.extract_filters("hola")
        assert filters == ExtractedFilters()

    def test_extract_price_with_k_suffix(self):
        filters = self.detector.extract_filters("hasta 70k")
        assert filters.max_price == 70000

    def test_extract_price_with_mil(self):
        filters = self.detector.extract_filters("60 mil dólares")
        assert filters.max_price == 60000

    def test_extract_price_venezuelan_separator(self):
        filters = self.detector.extract_filters("hasta 95.000")
        assert filters.max_price == 95000

    def test_detect_m2_ascii(self):
        """Variante ASCII de m² debe clasificar como FAQ de precios."""
        assert self.detector.detect("cuanto cuesta el m2 en pampatar") == IntentType.FAQ_PRICE_M2


# =============================================================================
# TESTS NUEVOS — Cubren bugs corregidos en fase 0.5
# =============================================================================

class TestPriceParsingFixes:
    """Bug crítico: re.sub con r'\1000' era ambiguo en Python."""

    def setup_method(self):
        self.detector = IntentDetector()

    def test_price_40k(self):
        filters = self.detector.extract_filters("busco propiedad en 40k")
        assert filters.max_price == 40000

    def test_price_40_k_with_space(self):
        filters = self.detector.extract_filters("quiero decir 40 k")
        assert filters.max_price == 40000

    def test_price_range_30_y_40_mil(self):
        filters = self.detector.extract_filters("entre 30 y 40 mil")
        assert filters.min_price == 30000
        assert filters.max_price == 40000

    def test_price_range_both_with_mil(self):
        filters = self.detector.extract_filters("entre 30 mil y 40 mil")
        assert filters.min_price == 30000
        assert filters.max_price == 40000


class TestZoneExtractionFixes:
    """Bug: matches cortos ganaban sobre largos (ej: 'Porlamar' vs 'Porlamar Centro')."""

    def setup_method(self):
        self.detector = IntentDetector()

    def test_zone_porlamar_centro(self):
        filters = self.detector.extract_filters("apartamento en Porlamar Centro")
        assert filters.zone == "Porlamar Centro"
        assert filters.municipality == Municipality.MARINO

    def test_zone_san_lorenzo(self):
        filters = self.detector.extract_filters("terreno en San Lorenzo")
        assert filters.zone == "San Lorenzo"
        assert filters.municipality == Municipality.MARINO

    def test_zone_apostadero(self):
        filters = self.detector.extract_filters("local en Apostadero")
        assert filters.zone == "Apostadero"
        assert filters.municipality == Municipality.MANEIRO

    def test_zone_altos_de_maneiro(self):
        filters = self.detector.extract_filters("apartamento en Altos de Maneiro")
        assert filters.zone == "Altos de Maneiro"
        assert filters.municipality == Municipality.MANEIRO

    def test_zone_guatamare(self):
        filters = self.detector.extract_filters("casa en Guatamare")
        assert filters.zone == "Guatamare"
        assert filters.municipality == Municipality.MARINO


class TestLeadExtractionFixes:
    """Bug: regex de teléfono no cubría WhatsApp, +58 con espacios, números standalone."""

    def setup_method(self):
        self.detector = IntentDetector()

    def test_lead_whatsapp(self):
        lead = self.detector.extract_lead_info("mi WhatsApp es +58 424 1112233")
        assert lead.phone == "584241112233"

    def test_lead_phone_country_code(self):
        lead = self.detector.extract_lead_info("mi número es +58 414 9876543")
        assert lead.phone == "584149876543"

    def test_lead_phone_standalone(self):
        lead = self.detector.extract_lead_info("0416-9990000")
        assert lead.phone == "4169990000"

    def test_lead_complete_with_whatsapp(self):
        msg = (
            "Hola, me llamo María Elena Rodríguez, "
            "mi WhatsApp es +58 424 1112233 y mi email es maria.r@gmail.com"
        )
        lead = self.detector.extract_lead_info(msg)
        assert lead.name == "María Elena Rodríguez"
        assert lead.phone == "584241112233"
        assert lead.email == "maria.r@gmail.com"
