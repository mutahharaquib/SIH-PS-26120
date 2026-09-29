"""Persistence: WellState snapshots and cards (SQLite via SQLAlchemy; the schema is
plain enough to move to TimescaleDB/Postgres unchanged)."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from core.state import Card, WellState


class Base(DeclarativeBase):
    pass


class StateRow(Base):
    __tablename__ = "well_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    well_id: Mapped[str] = mapped_column(String(32), index=True)
    t: Mapped[datetime] = mapped_column(DateTime, index=True)
    payload: Mapped[str] = mapped_column(Text)


class CardRow(Base):
    __tablename__ = "card"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    well_id: Mapped[str] = mapped_column(String(32), index=True)
    t: Mapped[datetime] = mapped_column(DateTime, index=True)
    kind: Mapped[str] = mapped_column(String(16))
    payload: Mapped[str] = mapped_column(Text)


class Store:
    def __init__(self, url: str = "sqlite:///:memory:") -> None:
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)

    def save_state(self, state: WellState) -> None:
        with Session(self.engine) as s:
            s.add(StateRow(well_id=state.well_id, t=state.t.replace(tzinfo=None), payload=state.model_dump_json()))
            s.commit()

    def save_card(self, well_id: str, card: Card) -> None:
        with Session(self.engine) as s:
            s.add(CardRow(well_id=well_id, t=card.t.replace(tzinfo=None), kind=card.kind, payload=card.model_dump_json()))
            s.commit()

    def history(self, well_id: str, t_from: datetime | None = None, t_to: datetime | None = None,
                limit: int = 5000) -> list[WellState]:
        with Session(self.engine) as s:
            stmt = select(StateRow).where(StateRow.well_id == well_id)
            if t_from:
                stmt = stmt.where(StateRow.t >= t_from.replace(tzinfo=None))
            if t_to:
                stmt = stmt.where(StateRow.t <= t_to.replace(tzinfo=None))
            rows = s.execute(stmt.order_by(StateRow.t.desc()).limit(limit)).scalars().all()
            return [WellState.model_validate(json.loads(r.payload)) for r in reversed(rows)]

    def cards(self, well_id: str, limit: int = 10) -> list[Card]:
        with Session(self.engine) as s:
            stmt = select(CardRow).where(CardRow.well_id == well_id).order_by(CardRow.id.desc()).limit(limit)
            rows = s.execute(stmt).scalars().all()
            return [Card.model_validate(json.loads(r.payload)) for r in reversed(rows)]

    def count_states(self, well_id: str) -> int:
        with Session(self.engine) as s:
            return len(s.execute(select(StateRow.id).where(StateRow.well_id == well_id)).all())
