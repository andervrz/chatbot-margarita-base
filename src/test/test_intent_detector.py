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
