"""Build the assistant's knowledge base from data/docs_manifest.json.

Steps (offline; never on a request path):
  1. download each PDF into data/docs/<id>.pdf (gitignored; skipped if already there unless
     --refresh) and verify it (a PDF whose first pages contain the entry's `expect` phrase);
     results go to data/docs/manifest.lock.json (sha256, pages, verified, reason);
  2. split verified documents into page-tagged chunks;
  3. embed them locally (fastembed, BAAI/bge-small-en-v1.5; model cached in .cache/models);
  4. write the local index data/docs/index/ and, with --db, the Postgres tables
     rag_documents / rag_chunks (pgvector; run `alembic upgrade head` first).

Usage (repo root, venv active):
    python scripts/ingest_docs.py --download-only
    python scripts/ingest_docs.py            # download + build local index
    python scripts/ingest_docs.py --db       # also write to Postgres (DATABASE_URL)
"""

import argparse
import sys
import time

import httpx

from app.ai.rag.corpus import (
    Chunk,
    DocEntry,
    LockEntry,
    chunk_document,
    load_manifest,
    lock_entry,
    verify,
    write_lock,
)
from app.ai.rag.embed import DEFAULT_MODEL, FastEmbedEmbedder
from app.ai.rag.store import InMemoryIndex
from app.core.config import REPO_ROOT, get_settings

MANIFEST = REPO_ROOT / "data" / "docs_manifest.json"
DOCS = REPO_ROOT / "data" / "docs"
INDEX = DOCS / "index"
MODELS = REPO_ROOT / ".cache" / "models"
HEADERS = {
    # Some AMC sites refuse requests without a browser-like user agent.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AIWealth-ingest/0.1",
    "Accept": "application/pdf,*/*",
}


def download(url: str) -> bytes:
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=90) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-download existing PDFs")
    ap.add_argument("--db", action="store_true", help="also write to Postgres")
    args = ap.parse_args(argv)

    manifest = load_manifest(MANIFEST)
    DOCS.mkdir(parents=True, exist_ok=True)
    locks: list[LockEntry] = []
    verified: list[tuple[DocEntry, list[str]]] = []
    for entry in manifest.documents:
        path = DOCS / f"{entry.id}.pdf"
        data: bytes | None = None
        try:
            if path.is_file() and not args.refresh:
                data = path.read_bytes()
            else:
                data = download(entry.url)
                time.sleep(1.0)  # be polite to the hosts
        except httpx.HTTPError as exc:
            reason = f"download failed: {type(exc).__name__}"
            locks.append(lock_entry(entry, False, reason, None, 0))
            print(f"  FAIL {entry.id}: download failed ({type(exc).__name__})")
            continue
        ok, reason, pages = verify(entry, data)
        locks.append(lock_entry(entry, ok, reason, data, len(pages)))
        status = "ok  " if ok else "FAIL"
        print(f"  {status} {entry.id}: {len(pages)} pages, {len(data):,} bytes - {reason}")
        if ok:
            path.write_bytes(data)
            verified.append((entry, pages))
        elif path.is_file():
            path.unlink()  # never keep an unverified file around
    write_lock(DOCS / "manifest.lock.json", locks)
    print(f"verified {len(verified)} of {len(manifest.documents)} documents")
    if args.download_only or not verified:
        return 0 if verified else 1

    chunks: list[Chunk] = []
    per_doc: dict[str, list[Chunk]] = {}
    for entry, pages in verified:
        doc_chunks = chunk_document(entry, pages)
        per_doc[entry.id] = doc_chunks
        chunks += doc_chunks
    print(f"{len(chunks):,} chunks; embedding with {DEFAULT_MODEL} ...", flush=True)
    t0 = time.perf_counter()
    embedder = FastEmbedEmbedder(DEFAULT_MODEL, MODELS)
    vectors = embedder.embed_documents([c.text for c in chunks])
    print(f"embedded in {time.perf_counter() - t0:.0f}s ({vectors.shape[1]} dims)")
    InMemoryIndex(chunks, vectors).save(INDEX, embedder.name)
    print(f"local index written to {INDEX}")

    if args.db:
        from app.ai.rag.store import PgVectorStore
        from app.db.session import get_engine

        if get_settings().database_url is None:
            print("DATABASE_URL is not set; skipping Postgres")
            return 1
        store = PgVectorStore(get_engine())
        offset = 0
        lock_by_id = {lk.id: lk for lk in locks}
        for entry, pages in verified:
            doc_chunks = per_doc[entry.id]
            vecs = vectors[offset : offset + len(doc_chunks)]
            offset += len(doc_chunks)
            lk = lock_by_id[entry.id]
            store.replace_document(
                {
                    "id": entry.id,
                    "title": entry.title,
                    "doc_type": entry.doc_type,
                    "fund_code": entry.fund_code,
                    "url": entry.url,
                    "sha256": lk.sha256,
                    "pages": len(pages),
                    "meta": {"model": embedder.name},
                },
                doc_chunks,
                vecs,
                embedder.name,
            )
        docs_n, chunks_n = store.counts()
        print(f"Postgres: {docs_n} documents, {chunks_n:,} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
