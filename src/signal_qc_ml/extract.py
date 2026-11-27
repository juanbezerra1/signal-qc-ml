"""Getting the table out of a labtrack-db database, and writing it down with where it came from."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, cast

import pandas as pd

from . import __version__
from . import columns as cols

DEFAULT_DSN = "postgresql://labtrack:labtrack@localhost:5432/labtrack"  # the docker-compose default of labtrack-db
FLOAT_DIGITS = 6  # the table is rounded to this many decimals, so that it is the same on every machine


class ExtractError(RuntimeError):
    """The table could not be read from the database."""


def get_dsn(dsn: str | None = None) -> str:
    """The DSN from the argument, from ``$LABTRACK_DSN`` (as labtrack-db does), or the docker-compose default."""
    return dsn or os.environ.get("LABTRACK_DSN") or DEFAULT_DSN


def query_text(name: str) -> str:
    """The text of one of the two queries that ship with the package."""
    return resources.files("signal_qc_ml").joinpath(name).read_text(encoding="utf-8").replace("\r\n", "\n")


def _connect(dsn: str) -> Any:
    try:
        import psycopg
    except ImportError as error:
        raise ExtractError(
            'reading a database needs psycopg: py -m pip install "signal-qc-ml[db]" (or psycopg[binary])'
        ) from error
    try:
        return psycopg.connect(dsn, autocommit=True)
    except psycopg.Error as error:
        raise ExtractError(f"could not connect to the database: {str(error).strip().splitlines()[0]}") from error


def _rows(conn: Any, sql: str) -> pd.DataFrame:
    cursor = conn.execute(sql)
    names = [column.name for column in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=names)


def fetch(dsn: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the two queries and return the table, and the facts about the database that go in its provenance."""
    with _connect(get_dsn(dsn)) as conn:
        try:
            recordings = _rows(conn, query_text("recordings.sql"))
            signals = _rows(conn, query_text("signals.sql"))
            latest = conn.execute("SELECT max(version) FROM public.schema_migrations").fetchone()
        except Exception as error:  # psycopg.Error and its subclasses: the schema is not the one of labtrack-db
            raise ExtractError(
                f"the queries failed, is this a labtrack-db database? {str(error).strip().splitlines()[0]}"
            ) from error
    facts = {"latest_migration": int(latest[0]) if latest and latest[0] is not None else None}
    return assemble(recordings, signals), facts


def assemble(recordings: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Join the two query results into one wide table: one row per recording, one column per summary of a feature."""
    wide = signals.pivot(index="recording_id", columns="feature", values=["mean", "sd"])
    pairs = cast("list[tuple[str, str]]", list(wide.columns))  # (statistic, feature)
    wide.columns = [f"{cols.SIGNAL_PREFIX}{feature}_{stat}" for stat, feature in pairs]
    wide = wide[sorted(wide.columns)]
    table = recordings.merge(wide, how="left", left_on="recording_id", right_index=True)
    table = table.sort_values("recording_id", kind="stable").reset_index(drop=True)
    table["recorded_on"] = table["recorded_on"].astype(str)
    floats = [name for name in table.columns if pd.api.types.is_float_dtype(table[name])]
    table[floats] = table[floats].round(FLOAT_DIGITS)
    return table


def write_table(table: pd.DataFrame, path: str | Path) -> str:
    """Write the table as UTF-8 with LF line endings (gzipped if the name ends in ``.gz``).

    Returns the SHA-256 of the CSV text itself, not of the file: a gzipped file is not the same bytes on every
    machine, the text inside it is.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = table.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if target.suffix == ".gz":
        with (
            target.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz,
        ):
            gz.write(data)
    else:
        target.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def meta_path(path: str | Path) -> Path:
    """Where the provenance of a table is: ``recordings-small.csv`` and ``recordings-small.csv.gz`` share one file."""
    target = Path(path)
    name = target.name.removesuffix(".gz").removesuffix(".csv")
    return target.with_name(f"{name}.meta.json")


def sha256_of_text(path: str | Path) -> str:
    """The SHA-256 of the CSV text of a table file, whether it is gzipped or not."""
    target = Path(path)
    data = gzip.decompress(target.read_bytes()) if target.suffix == ".gz" else target.read_bytes()
    return hashlib.sha256(data).hexdigest()


def verify(path: str | Path) -> bool | None:
    """Whether a table is still the one its provenance file describes: None if there is no provenance file."""
    meta = meta_path(path)
    if not meta.is_file():
        return None
    recorded = json.loads(meta.read_text(encoding="utf-8"))["table"]["sha256"]
    return bool(recorded == sha256_of_text(path))


def provenance(table: pd.DataFrame, path: str | Path, sha256: str, facts: dict[str, Any]) -> dict[str, Any]:
    """What a reader needs to know about a table to trust it: how it was made, and a hash to check it by."""
    reviewed = table[cols.TARGET] != cols.PENDING
    return {
        "tool": f"signal-qc-ml {__version__}",
        "extracted_on": date.today().isoformat(),
        "source": {"labtrack_db_latest_migration": facts.get("latest_migration")},
        "queries": {
            name: hashlib.sha256(query_text(name).encode("utf-8")).hexdigest()
            for name in ("recordings.sql", "signals.sql")
        },
        "table": {
            "file": Path(path).name,
            "sha256": sha256,
            "rows": len(table),
            "reviewed": int(reviewed.sum()),
            "pending": int((~reviewed).sum()),
            "participants": int(table[cols.GROUP].nunique()),
            "columns": len(table.columns),
        },
    }


def snapshot(dsn: str | None, path: str | Path) -> dict[str, Any]:
    """Extract the table, write it to ``path`` and where it came from next to it (``recordings-small.meta.json``)."""
    table, facts = fetch(dsn)
    sha256 = write_table(table, path)
    meta = provenance(table, path, sha256, facts)
    meta_path(path).write_text(json.dumps(meta, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    return meta
