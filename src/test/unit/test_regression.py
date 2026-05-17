"""
Tests de regresión para bugs corregidos en fase 0.5.
Verifica: precios con espacio/k/mil, extracción de zona, captura de lead.
"""

import pytest

from src.application.intent import ExtractedFilters, IntentDetector, IntentType, PartialLead
from src.domain.models import Municipality, PropertyType


class TestPriceParsingRegression:
    """Bug A: precios con espacio, 'k', 'mil', rangos asimétricos."""

    def setup_method(self):
        self.detector = IntentDetector()

    def test_price_40k_without_prefix(self):
        """'busco propiedad en 40k' debe extraer max_price=40000."""
        filters = self.detector.extract_filters("busco propiedad en 40k")
        assert filters.max_price == 40000, f"Expected 40000, got {filters.max_price}"

    def test_price_40_k_with_space(self):
        """'quiero decir 40 k' debe extraer max_price=40000."""
        filters = self.detector.extract_filters("quiero decir 40 k")
        assert filters.max_price == 40000, f"Expected 40000, got {filters.max_price}"

    def test_price_40_mil(self):
        """'40 mil' debe extraer max_price=40000."""
        filters = self.detector.extract_filters("40 mil")
        assert filters.max_price == 40000, f"Expected 40000, got {filters.max_price}"

    def test_price_range_30_y_40_mil(self):
        """'entre 30 y 40 mil' debe extraer min=30000, max=40000."""
        filters = self.detector.extract_filters("entre 30 y 40 mil")
        assert filters.min_price == 30000, f"Expected min=30000, got {filters.min_price}"
        assert filters.max_price == 40000, f"Expected max=40000, got {filters.max_price}"

    def test_price_range_both_with_mil(self):
        """'entre 30 mil y 40 mil' debe extraer min=30000, max=40000."""
        filters = self.detector.extract_filters("entre 30 mil y 40 mil")
        assert filters.min_price == 30000
        assert filters.max_price == 40000

    def test_price_70k_suffix(self):
        """'hasta 70k' debe seguir funcionando (regresión de test anterior)."""
        filters = self.detector.extract_filters("hasta 70k")
        assert filters.max_price == 70000

    def test_price_60_mil(self):
        """'60 mil dólares' debe seguir funcionando (regresión de test anterior)."""
        filters = self.detector.extract_filters("60 mil dólares")
        assert filters.max_price == 60000

    def test_price_venezuelan_separator(self):
        """'hasta 95.000' debe seguir funcionando (regresión de test anterior)."""
        filters = self.detector.extract_filters("hasta 95.000")
        assert filters.max_price == 95000


class TestZoneExtractionRegression:
    """Bug B: zone nunca se extraía, solo municipality."""

    def setup_method(self):
        self.detector = IntentDetector()

    def test_extract_zone_pampatar(self):
        """'casa en pampatar' debe extraer zone='Pampatar' y municipality=MANEIRO."""
        filters = self.detector.extract_filters("casa en pampatar")
        assert filters.zone == "Pampatar", f"Expected zone='Pampatar', got {filters.zone}"
        assert filters.municipality == Municipality.MANEIRO

    def test_extract_zone_porlamar_centro(self):
        """'apartamento en Porlamar Centro' debe extraer zona específica."""
        filters = self.detector.extract_filters("apartamento en Porlamar Centro")
        assert filters.zone == "Porlamar Centro"
        assert filters.municipality == Municipality.MARINO

    def test_extract_zone_playa_el_agua(self):
        """'terreno en playa el agua' debe extraer zona turística."""
        filters = self.detector.extract_filters("terreno en playa el agua")
        assert filters.zone == "Playa El Agua"
        assert filters.municipality == Municipality.ANTOLIN_DEL_CAMPO

    def test_extract_zone_sabanamar(self):
        """'casa en Sabanamar' debe extraer zona de Mariño."""
        filters = self.detector.extract_filters("casa en Sabanamar")
        assert filters.zone == "Sabanamar"
        assert filters.municipality == Municipality.MARINO

    def test_multiple_filters_with_zone(self):
        """Filtros múltiples incluyendo zona específica."""
        filters = self.detector.extract_filters(
            "apartamento en Playa El Agua hasta 100000 con 2 habitaciones"
        )
        assert filters.property_type == PropertyType.APARTAMENTO
        assert filters.zone == "Playa El Agua"
        assert filters.municipality == Municipality.ANTOLIN_DEL_CAMPO
        assert filters.max_price == 100000
        assert filters.min_bedrooms == 2


class TestLeadExtractionRegression:
    """Bug C: CAPTURE_LEAD detectado pero sin extracción ni persistencia."""

    def setup_method(self):
        self.detector = IntentDetector()

    def test_detect_capture_lead_intent(self):
        """'mi nombre es Carlos' debe detectar CAPTURE_LEAD."""
        intent = self.detector.detect("mi nombre es Carlos")
        assert intent == IntentType.CAPTURE_LEAD

    def test_extract_lead_name(self):
        """Extrae nombre de 'mi nombre es Carlos Rodríguez'."""
        lead = self.detector.extract_lead_info("mi nombre es Carlos Rodríguez")
        assert lead.name == "Carlos Rodríguez"

    def test_extract_lead_name_me_llamo(self):
        """Extrae nombre de 'me llamo María González'."""
        lead = self.detector.extract_lead_info("me llamo María González")
        assert lead.name == "María González"

    def test_extract_lead_phone(self):
        """Extrae teléfono de 'mi teléfono es 0412-1234567'."""
        lead = self.detector.extract_lead_info("mi teléfono es 0412-1234567")
        assert lead.phone == "04121234567"

    def test_extract_lead_email(self):
        """Extrae email de 'mi correo es carlos@test.com'."""
        lead = self.detector.extract_lead_info("mi correo es carlos@test.com")
        assert lead.email == "carlos@test.com"

    def test_extract_lead_complete(self):
        """Extrae todos los datos de un mensaje completo."""
        msg = "Hola, me llamo Juan Pérez, mi teléfono es 0414-9876543 y mi email es juan@email.com"
        lead = self.detector.extract_lead_info(msg)
        assert lead.name == "Juan Pérez"
        assert lead.phone == "04149876543"
        assert lead.email == "juan@email.com"

    def test_extract_lead_partial(self):
        """Maneja mensaje con solo nombre."""
        lead = self.detector.extract_lead_info("me llamo Ana")
        assert lead.name == "Ana"
        assert lead.phone is None
        assert lead.email is None


class TestIntentDetectionRegression:
    """Regresiones: detect debe seguir funcionando para casos existentes."""

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
