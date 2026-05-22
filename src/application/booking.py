"""
Servicio de agendamiento.
SRP: ChatService decide CUÁNDO agendar, BookingService decide CÓMO guardar y notificar.
"""

from datetime import datetime, timezone

import structlog

from src.domain.models import Appointment, AppointmentStatus, Lead
from src.domain.exceptions import DomainError
from src.infrastructure.db import Database
from src.infrastructure.whatsapp import WhatsAppClient

logger = structlog.get_logger()


class BookingService:
    """
    Gestiona citas de visita a propiedades.
    Reglas de negocio:
    - Todas las citas nacen con status='pending'
    - NUNCA confirmar automáticamente (solo humanos pueden confirmar)
    - Notificar al agente por WhatsApp cuando se crea una cita
    """

    def __init__(self, db: Database, whatsapp: WhatsAppClient) -> None:
        self.db = db
        self.whatsapp = whatsapp

    async def create_appointment(
        self,
        session_id: str,
        property_id: int | None,
        lead_id: int | None,
        requested_date: str | None = None,
        requested_time: str | None = None,
        notes: str = "",
    ) -> Appointment:
        """
        Crea una cita pendiente y notifica al agente.
        """
        cursor = await self.db.execute(
            """
            INSERT INTO appointments (session_id, property_id, lead_id, requested_date, requested_time, status, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                property_id,
                lead_id,
                requested_date,
                requested_time,
                AppointmentStatus.PENDING.value,
                notes,
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        await self.db.commit()

        appointment = Appointment(
            id=cursor.lastrowid,
            session_id=session_id,
            property_id=property_id,
            lead_id=lead_id,
            requested_date=requested_date,
            requested_time=requested_time,
            status=AppointmentStatus.PENDING,
            notes=notes,
        )

        logger.info(
            "appointment_created",
            appointment_id=appointment.id,
            session_id=session_id,
            status=AppointmentStatus.PENDING.value,
        )

        # Notificar al agente por WhatsApp
        await self._notify_agent(appointment)

        return appointment

    async def get_pending_appointments(self) -> list[Appointment]:
        """SELECT citas pendientes para dashboard/admin."""
        rows = await self.db.fetchall(
            "SELECT * FROM appointments WHERE status = ? ORDER BY created_at DESC",
            (AppointmentStatus.PENDING.value,),
        )
        return [self._row_to_appointment(r) for r in rows]

    async def confirm_appointment(self, appointment_id: int) -> Appointment:
        """UPDATE status='confirmed' — llamado por humano/agente."""
        await self.db.execute(
            """
            UPDATE appointments SET status = ?, updated_at = ? WHERE id = ?
            """,
            (
                AppointmentStatus.CONFIRMED.value,
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                appointment_id,
            ),
        )
        await self.db.commit()

        logger.info("appointment_confirmed", appointment_id=appointment_id)
        return await self._get_by_id(appointment_id)

    async def cancel_appointment(self, appointment_id: int) -> Appointment:
        """UPDATE status='cancelled'."""
        await self.db.execute(
            """
            UPDATE appointments SET status = ?, updated_at = ? WHERE id = ?
            """,
            (
                AppointmentStatus.CANCELLED.value,
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                appointment_id,
            ),
        )
        await self.db.commit()

        logger.info("appointment_cancelled", appointment_id=appointment_id)
        return await self._get_by_id(appointment_id)

    async def get_appointment_by_session(self, session_id: str) -> Appointment | None:
        """Recupera cita más reciente de una sesión."""
        row = await self.db.fetchone(
            "SELECT * FROM appointments WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        if not row:
            return None
        return self._row_to_appointment(row)

    # ---------- Internos ----------

    async def _notify_agent(self, appointment: Appointment) -> None:
        """Construye resumen y envía por WhatsApp."""
        # Recuperar datos del lead si existe
        lead_info = ""
        if appointment.lead_id:
            row = await self.db.fetchone(
                "SELECT name, phone, email FROM leads WHERE id = ?", (appointment.lead_id,)
            )
            if row:
                lead_info = (
                    f"Lead: {row['name'] or 'N/A'}\n"
                    f"Tel: {row['phone'] or 'N/A'}\n"
                    f"Email: {row['email'] or 'N/A'}"
                )

        # Recuperar datos de propiedad si existe
        property_info = ""
        if appointment.property_id:
            row = await self.db.fetchone(
                "SELECT title, zone, price_usd FROM properties WHERE id = ?", (appointment.property_id,)
            )
            if row:
                property_info = (
                    f"Propiedad: {row['title']}\n"
                    f"Zona: {row['zone']}\n"
                    f"Precio: ${row['price_usd']:,.0f}"
                )

        summary = (
            f"📅 NUEVA CITA PENDIENTE #{appointment.id}\n\n"
            f"{property_info}\n\n"
            f"{lead_info}\n\n"
            f"Horario solicitado: {appointment.requested_date or 'N/A'} {appointment.requested_time or 'N/A'}\n"
            f"Notas: {appointment.notes or 'N/A'}\n\n"
            f"⚠️ CONFIRMAR MANUALMENTE"
        )

        await self.whatsapp.send_lead_notification(summary)

    async def _get_by_id(self, appointment_id: int) -> Appointment:
        row = await self.db.fetchone("SELECT * FROM appointments WHERE id = ?", (appointment_id,))
        if not row:
            raise DomainError(f"Cita {appointment_id} no encontrada")
        return self._row_to_appointment(row)

    def _row_to_appointment(self, row) -> Appointment:
        return Appointment(
            id=row["id"],
            session_id=row["session_id"],
            property_id=row["property_id"],
            lead_id=row["lead_id"],
            requested_date=row["requested_date"],
            requested_time=row["requested_time"],
            status=AppointmentStatus(row["status"]),
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )