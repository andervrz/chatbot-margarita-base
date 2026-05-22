"""
Repositorio de propiedades.
Regla de oro: si no está en SQLite, el bot no lo inventa.
"""

import json

import structlog

from src.domain.exceptions import PropertyNotFound
from src.domain.models import Municipality, Property, PropertyStatus, PropertyType
from src.infrastructure.db import Database

logger = structlog.get_logger()


class PropertyRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------- Búsqueda SQL estructurada ----------

    async def search_exact(
        self,
        municipality: Municipality | None = None,
        zone: str | None = None,
        property_type: PropertyType | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        min_bedrooms: int | None = None,
        status: PropertyStatus = PropertyStatus.AVAILABLE,
        limit: int = 10,
        # NUEVO Fase 1: Filtros booleanos
        has_ocean_view: bool | None = None,
        is_furnished: bool | None = None,
        has_pool: bool | None = None,
        has_parking: bool | None = None,
        has_security: bool | None = None,
        has_generator: bool | None = None,
        has_water_tank: bool | None = None,
        has_ac: bool | None = None,
        is_new_construction: bool | None = None,
        has_balcony: bool | None = None,
        is_gated_community: bool | None = None,
    ) -> list[Property]:
        """
        Búsqueda por filtros estructurados incluyendo columnas booleanas.
        """
        conditions = ["status = ?"]
        params: list = [status.value]

        if municipality:
            conditions.append("municipality = ?")
            params.append(municipality.value)
        if zone:
            conditions.append("zone = ?")
            params.append(zone)
        if property_type:
            conditions.append("type = ?")
            params.append(property_type.value)
        if min_price is not None:
            conditions.append("price_usd >= ?")
            params.append(min_price)
        if max_price is not None:
            conditions.append("price_usd <= ?")
            params.append(max_price)
        if min_bedrooms is not None:
            conditions.append("bedrooms >= ?")
            params.append(min_bedrooms)

        # NUEVO Fase 1: Filtros booleanos (SQLite usa 0/1)
        bool_filters = {
            "has_ocean_view": has_ocean_view,
            "is_furnished": is_furnished,
            "has_pool": has_pool,
            "has_parking": has_parking,
            "has_security": has_security,
            "has_generator": has_generator,
            "has_water_tank": has_water_tank,
            "has_ac": has_ac,
            "is_new_construction": is_new_construction,
            "has_balcony": has_balcony,
            "is_gated_community": is_gated_community,
        }
        for col, val in bool_filters.items():
            if val is not None:
                conditions.append(f"{col} = ?")
                params.append(1 if val else 0)

        sql = f"SELECT * FROM properties WHERE {' AND '.join(conditions)} LIMIT ?"
        params.append(limit)

        rows = await self.db.fetchall(sql, params)
        logger.debug("property_search_sql", count=len(rows), filters=len(conditions) - 1)
        return [self._row_to_property(r) for r in rows if r]

    # ---------- Búsqueda vectorial (fallback) ----------

    async def search_vector(
        self,
        query_embedding: list[float],
        limit: int = 5,
        max_distance: float = 0.55,
    ) -> list[Property]:
        rows = await self.db.vec_search(
            table="property_embeddings",
            vector=query_embedding,
            vector_column="description_embedding",
            limit=limit,
            max_distance=max_distance,
        )
        if not rows:
            return []

        property_ids = [r["property_id"] for r in rows]
        placeholders = ",".join("?" for _ in property_ids)
        sql = f"""
            SELECT * FROM properties
            WHERE id IN ({placeholders}) AND status = ?
        """
        prop_rows = await self.db.fetchall(sql, property_ids + [PropertyStatus.AVAILABLE.value])

        props_by_id = {r["id"]: self._row_to_property(r) for r in prop_rows}
        results = []
        missing_ids = []
        for r in rows:
            pid = r["property_id"]
            if pid in props_by_id:
                results.append(props_by_id[pid])
            else:
                missing_ids.append(pid)

        if missing_ids:
            logger.warning(
                "property_vector_orphans",
                missing_ids=missing_ids,
                hint="Desincronización entre property_embeddings y properties",
            )

        logger.debug("property_search_vector", count=len(results), requested=len(rows))
        return results

    # ---------- CRUD básico ----------

    async def get_by_id(self, property_id: int) -> Property:
        row = await self.db.fetchone(
            "SELECT * FROM properties WHERE id = ?", (property_id,)
        )
        if not row:
            raise PropertyNotFound(f"Propiedad {property_id} no existe en base de datos")
        return self._row_to_property(row)

    async def create(self, property: Property) -> Property:
        sql = """
        INSERT INTO properties (
            title, municipality, zone, type, price_usd, bedrooms, bathrooms,
            area_m2, description, features, status, contact_phone, contact_email,
            has_ocean_view, is_furnished, has_pool, has_parking, has_security,
            has_generator, has_water_tank, has_ac, is_new_construction,
            has_balcony, is_gated_community
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            property.title,
            property.municipality.value,
            property.zone,
            property.type.value,
            property.price_usd,
            property.bedrooms,
            property.bathrooms,
            property.area_m2,
            property.description,
            json.dumps(property.features),
            property.status.value,
            property.contact_phone,
            property.contact_email,
            int(property.has_ocean_view),
            int(property.is_furnished),
            int(property.has_pool),
            int(property.has_parking),
            int(property.has_security),
            int(property.has_generator),
            int(property.has_water_tank),
            int(property.has_ac),
            int(property.is_new_construction),
            int(property.has_balcony),
            int(property.is_gated_community),
        )
        cursor = await self.db.execute(sql, params)
        await self.db.commit()
        property.id = cursor.lastrowid
        return property

    # ---------- Fuente Única de Verdad para Zonas ----------

    async def get_zone_index(self) -> dict[str, tuple[Municipality, str]]:
        rows = await self.db.fetchall(
            "SELECT DISTINCT zone, municipality FROM properties WHERE status = ?",
            (PropertyStatus.AVAILABLE.value,)
        )

        zone_index: dict[str, tuple[Municipality, str]] = {}
        for row in rows:
            try:
                municipality = Municipality(row["municipality"])
                zone_name = row["zone"]
                key = zone_name.lower()
                zone_index[key] = (municipality, zone_name)
            except ValueError:
                logger.warning(
                    "zone_index_skip_invalid_municipality",
                    municipality=row["municipality"],
                    zone=row["zone"],
                )
                continue

        logger.info("zone_index_built", zones=len(zone_index))
        return zone_index

    # ---------- Helpers ----------

    def _row_to_property(self, row) -> Property | None:
        features = []
        if row["features"]:
            try:
                features = json.loads(row["features"])
            except json.JSONDecodeError:
                features = []

        try:
            keys = list(row.keys())

            # NUEVO Fase 1: Mapear columnas booleanas (SQLite retorna 0/1)
            def _bool(col: str) -> bool:
                return row[col] == 1 if col in keys and row[col] is not None else False

            return Property(
                id=row["id"],
                title=row["title"],
                municipality=Municipality(row["municipality"]),
                zone=row["zone"],
                type=PropertyType(row["type"]),
                price_usd=row["price_usd"],
                bedrooms=row["bedrooms"],
                bathrooms=row["bathrooms"],
                area_m2=row["area_m2"],
                description=row["description"],
                features=features,
                status=PropertyStatus(row["status"]),
                contact_phone=row["contact_phone"],
                contact_email=row["contact_email"],
                # NUEVO Fase 1
                has_ocean_view=_bool("has_ocean_view"),
                is_furnished=_bool("is_furnished"),
                has_pool=_bool("has_pool"),
                has_parking=_bool("has_parking"),
                has_security=_bool("has_security"),
                has_generator=_bool("has_generator"),
                has_water_tank=_bool("has_water_tank"),
                has_ac=_bool("has_ac"),
                is_new_construction=_bool("is_new_construction"),
                has_balcony=_bool("has_balcony"),
                is_gated_community=_bool("is_gated_community"),
                created_at=row["created_at"] if "created_at" in keys else None,
                updated_at=row["updated_at"] if "updated_at" in keys else None,
            )
        except (ValueError, KeyError) as exc:
            logger.warning(
                "property_row_corrupt",
                property_id=row.get("id") if hasattr(row, "get") else row["id"],
                error=str(exc),
                row_keys=keys,
            )
            return None