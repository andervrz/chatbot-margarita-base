"""
Repositorio de propiedades.
Regla de oro: si no está en SQLite, el bot no lo inventa.
Búsqueda exacta primero; fallback vectorial si no hay resultados estructurados.
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
    ) -> list[Property]:
        """
        Búsqueda por filtros estructurados.
        Cero alucinación: solo retorna lo que existe en la tabla properties.
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

        sql = f"SELECT * FROM properties WHERE {' AND '.join(conditions)} LIMIT ?"
        params.append(limit)

        rows = await self.db.fetchall(sql, params)
        logger.debug("property_search_sql", count=len(rows), filters=len(conditions) - 1)
        return [self._row_to_property(r) for r in rows]

    # ---------- Búsqueda vectorial (fallback) ----------

    async def search_vector(
        self,
        query_embedding: list[float],
        limit: int = 5,
        max_distance: float = 0.55,  # ~cosine > 0.85
    ) -> list[Property]:
        """
        Fallback cuando la búsqueda SQL no retorna resultados.
        Compara el embedding de la consulta contra descripciones de propiedades.
        """
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

        # Mantener orden de relevancia vectorial
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
            area_m2, description, features, status, contact_phone, contact_email
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        )
        cursor = await self.db.execute(sql, params)
        await self.db.commit()
        property.id = cursor.lastrowid
        return property

    # ---------- Helpers ----------

    def _row_to_property(self, row) -> Property | None:
        """
        Convierte una fila de SQLite a Property.
        Retorna None si los datos están corruptos (enum inválido, etc.).
        """
        features = []
        if row["features"]:
            try:
                features = json.loads(row["features"])
            except json.JSONDecodeError:
                features = []

        try:
            # sqlite3.Row no tiene .get(), usar acceso directo con verificación
            keys = list(row.keys())
            
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
                # Campos de auditoría opcionales
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
