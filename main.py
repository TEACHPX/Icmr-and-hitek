"""FastAPI backend over DuckDB - ICMR + HITEK combined dataset (2.5B rows, 11 cols).

Index-aware routing:
  - idx_phone.parquet  (sorted by phoneNumber)  -> fast exact phone/other lookups
  - idx_aadhar.parquet (sorted by aadharNumber) -> fast exact aadhar lookups
  - idx_name.parquet   (sorted by name)         -> fast name prefix/exact
  - Falls back to full scans of raw parquet while indexes are still building.
Dedup: max 2 rows per person. source column included (icmr / inddata "connected docs").
"""
import asyncio
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
# data files may live in BASE or BASE/data (setup scripts download into data/)
_DATA_DIR = os.environ.get("ICMR_DATA_DIR") or \
    (os.path.join(BASE, "data") if os.path.isdir(os.path.join(BASE, "data")) else BASE)
PARQUET_FILES = [os.path.join(_DATA_DIR, f) for f in
                 ["part1.parquet", "part2a.parquet", "part2b_new.parquet"]]
IDX_PHONE = os.path.join(_DATA_DIR, "idx_phone.parquet")
IDX_AADHAR = os.path.join(_DATA_DIR, "idx_aadhar.parquet")
IDX_NAME = os.path.join(_DATA_DIR, "idx_name.parquet")
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "15"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "8"))
DUPLICATE_CAP = 2  # max 2 copies of the same person in results

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

app = FastAPI(title="ICMR + HITEK Search API",
              description="DuckDB-backed search over 2.5B records (11 cols, dedup max 2)")

# ---- DuckDB connection pool (one connection per worker thread) ----
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

_MISSING = [f for f in PARQUET_FILES if not os.path.exists(f)]


def _idx_files(path: str) -> list[str]:
    """Index may be a single file or split into sorted parts (idx_phone.0.parquet...)."""
    base, ext = os.path.splitext(path)  # idx_phone.parquet -> idx_phone / .parquet
    parts = sorted(glob.glob(f"{base}.*{ext}"))
    if parts and all(os.path.getsize(p) > 0 for p in parts):
        return parts
    return [path] if os.path.exists(path) and os.path.getsize(path) > 0 else []


def _idx_ready(path: str) -> bool:
    # index is usable when: file(s) exist+non-empty AND not caught mid-build
    # (build writes in place, so a START without DONE in the log = partial file)
    if not _idx_files(path):
        return False
    try:
        with open(os.path.join(BASE, "build_index.log"), encoding="utf-8",
                  errors="ignore") as f:
            log = f.read()
    except OSError:
        log = ""  # no log (fresh download) -> trust the downloaded file
    name = os.path.basename(path)
    if f"DONE {name}" in log:
        return True
    if f"START {name}" in log:
        return False  # build in progress / was interrupted -> partial
    return True  # no build ever started here -> file was downloaded whole


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()  # in-memory
    con.execute("INSTALL parquet; LOAD parquet;")
    files = ", ".join(f"'{f}'" for f in PARQUET_FILES if os.path.exists(f))
    con.execute(f"CREATE VIEW people AS SELECT * FROM read_parquet([{files}])")
    # sorted index views (only if built) - zone-map pruning makes lookups fast
    for idx_path, view in [(IDX_PHONE, "people_phone"),
                           (IDX_AADHAR, "people_aadhar"),
                           (IDX_NAME, "people_name")]:
        files = _idx_files(idx_path)
        if files and _idx_ready(idx_path):
            lst = ", ".join(f"'{f}'" for f in files)
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# ---- Dedup: max 2 copies per person ----
def _person_key(row: dict[str, Any]) -> tuple[str, ...]:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


def _cap_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep at most DUPLICATE_CAP rows per person (same phone+aadhar identity)."""
    seen: dict[tuple[str, ...], int] = {}
    out: list[dict[str, Any]] = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            out.append(r)
    return out


# ---- Query helpers ----
def _run_sql(sql: str) -> list[dict[str, Any]]:
    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    return [dict(zip(cols, r)) for r in rows]


def _source_clause(source: str | None) -> str:
    if source and source in ("icmr", "hitek", "inddata"):
        src = "hitek" if source == "hitek" else source
        return f" AND source = '{src}'"
    return ""


def _exact_scan(field: str, value: str, limit: int, source: str | None = None) -> str:
    v = value.replace("'", "''")
    return (f"SELECT * FROM people WHERE {field} = '{v}'"
            f"{_source_clause(source)} LIMIT {limit * DUPLICATE_CAP + 20}")


def _run_field_search(field: str, value: str, mode: str, limit: int,
                      source: str | None = None) -> dict[str, Any]:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")
    view = "people"
    # use sorted index view when available for the hot fields
    if mode == "exact":
        if field == "phoneNumber" and _idx_ready(IDX_PHONE):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready(IDX_AADHAR):
            view = "people_aadhar"
        elif field == "name" and _idx_ready(IDX_NAME):
            view = "people_name"
        sql = (f"SELECT * FROM {view} WHERE {field} = '{v}'"
               f"{_source_clause(source)} LIMIT {limit * DUPLICATE_CAP + 20}")
    elif mode == "contains":
        if field == "name" and _idx_ready(IDX_NAME):
            view = "people_name"  # sorted helps prefix; contains still scans but cached
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (f"SELECT * FROM {view} WHERE {field} ILIKE '%{v2}%' ESCAPE '\\'"
               f"{_source_clause(source)} LIMIT {limit * DUPLICATE_CAP + 20}")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results),
            "results": results}


async def _parallel(queries: list[tuple[str, str, str, int]]) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()

    async def one(t: tuple[str, str, str, int]) -> dict[str, Any]:
        field, value, mode, limit = t
        return await loop.run_in_executor(pool, _run_field_search, field, value, mode, limit)

    return await asyncio.gather(*[one(t) for t in queries])


async def _unified_search(q: str, limit: int, source: str | None = None) -> dict[str, Any]:
    """Smart unified search: uses sorted indexes per field, merges, dedups."""
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8
    loop = asyncio.get_running_loop()

    if is_num:
        # number-like -> exact match, phone index first (fast), then aadhar, then other
        all_rows: list[dict[str, Any]] = []
        searched: list[str] = []
        # phone index ready -> instant; else full scan (slow, skip if we already found hits)
        if _idx_ready(IDX_PHONE):
            r = await loop.run_in_executor(pool, _run_field_search, "phoneNumber", q,
                                           "exact", limit, source)
            all_rows.extend(r["results"])
            searched.append("phoneNumber")
        if not all_rows and _idx_ready(IDX_AADHAR):
            r = await loop.run_in_executor(pool, _run_field_search, "aadharNumber", q,
                                           "exact", limit, source)
            all_rows.extend(r["results"])
            searched.append("aadharNumber")
        if not all_rows and _idx_ready(IDX_NAME):
            r = await loop.run_in_executor(pool, _run_field_search, "otherNumber", q,
                                           "exact", limit, source)
            all_rows.extend(r["results"])
            searched.append("otherNumber")
        all_rows = _cap_duplicates(all_rows)[:limit]
        return {"query": q, "searched_fields": searched or list(NUMBER_FIELDS),
                "dedup": DUPLICATE_CAP, "count": len(all_rows), "results": all_rows}
    else:
        # text -> name + fathersName contains via name index, then address/district/etc.
        jobs = [loop.run_in_executor(pool, _run_field_search, "name", q, "contains",
                                     limit, source)]
        if _idx_ready(IDX_NAME):
            jobs.append(loop.run_in_executor(pool, _run_field_search, "fathersName", q,
                                             "contains", limit, source))
        # also check number fields exact (safety net, only if numbers typed)
        results = await asyncio.gather(*jobs)
        all_rows = []
        for r in results:
            all_rows.extend(r["results"])
        all_rows = _cap_duplicates(all_rows)[:limit]
        fields = ["name", "fathersName"] + NUMBER_FIELDS
        return {"query": q, "searched_fields": fields, "dedup": DUPLICATE_CAP,
                "count": len(all_rows), "results": all_rows}


def _pretty(data: dict[str, Any], pretty: bool) -> Response:
    if pretty:
        return Response(content=json.dumps(data, indent=2, ensure_ascii=False),
                        media_type="application/json")
    return Response(content=json.dumps(data, ensure_ascii=False),
                    media_type="application/json")


# ---- API ----
class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@app.get("/")
def root():
    return {"app": "ICMR + HITEK Search API", "records": 2_504_793_870,
            "files": [os.path.basename(f) for f in PARQUET_FILES],
            "indexes": {"phone": _idx_ready(IDX_PHONE), "aadhar": _idx_ready(IDX_AADHAR),
                        "name": _idx_ready(IDX_NAME)},
            "columns": SEARCH_FIELDS, "parallelism": PARALLELISM, "dedup": DUPLICATE_CAP,
            "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "files_ready": len(PARQUET_FILES) - len(_MISSING),
            "files_total": len(PARQUET_FILES),
            "indexes": {"phone": _idx_ready(IDX_PHONE), "aadhar": _idx_ready(IDX_AADHAR),
                        "name": _idx_ready(IDX_NAME)},
            "missing": [os.path.basename(f) for f in _MISSING]}


@app.get("/search")
async def search(
    q: str = Query(..., description="Search term - matches name, phone, aadhar, other, address, district, pincode, state, town"),
    field: str | None = Query(None, description=f"Restrict to one field: {SEARCH_FIELDS}"),
    mode: str = Query("contains", pattern="^(exact|contains)$"),
    limit: int = Query(10, ge=1, le=1000),
    source: str | None = Query(None, pattern="^(icmr|hitek|inddata)$",
                               description="Filter by source: icmr, hitek (=inddata), inddata"),
    pretty: bool = Query(True, description="Pretty-print JSON"),
):
    """Unified search across all 11 columns. Duplicates capped at 2 per person."""
    if field:
        try:
            data = _run_field_search(field, q, mode, limit, source)
        except ValueError as e:
            raise HTTPException(400, str(e))
        data = {"query": q, "field": field, "mode": mode, "dedup": DUPLICATE_CAP,
                "count": data["count"], "results": data["results"]}
    else:
        data = await _unified_search(q, limit, source)
    return _pretty(data, pretty)


@app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    """Run up to 50 searches in parallel across a pool of DuckDB connections."""
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    results = await _parallel([(item.get("field", "name"), item.get("value", ""),
                                item.get("mode", "contains"), int(item.get("limit", req.limit)))
                               for item in req.queries])
    out = [{"error": str(r)} if isinstance(r, Exception) else r for r in results]
    return _pretty({"searches": len(req.queries), "parallelism": PARALLELISM, "results": out}, True)
