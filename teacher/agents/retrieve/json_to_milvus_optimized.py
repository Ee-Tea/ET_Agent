
import os
import re
import json
import uuid
import math
import hashlib
import logging
from typing import List, Dict, Any, Iterable, Tuple, Optional

from pymilvus import (
    connections, utility, Collection, FieldSchema, CollectionSchema, DataType
)
from sentence_transformers import SentenceTransformer


# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("json_to_milvus_optimized")


# ---------------------------
# Helpers
# ---------------------------

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(s: str) -> str:
    """Lightweight cleanup for OCR/bullet-heavy content."""
    if not s:
        return ""
    # unify line endings and spaces
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    # normalize bullets to newlines
    s = re.sub(r"(?:\s*[•●·▪▶►\-–—]\s+)", "\n- ", s)
    # collapse multiple newlines
    s = re.sub(r"\n{3,}", "\n\n", s)
    # trim boilerplate cruft characters
    s = s.strip(" \n\t-·•●▶►")
    # collapse long dashes around words (OCR noise)
    s = re.sub(r"\s*-\s*", "- ", s)
    return s.strip()


def good_enough(text: str, min_chars: int, min_tokens: int, tok) -> bool:
    if not text or len(text) < min_chars:
        return False
    if min_tokens > 0:
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) < min_tokens:
            return False
    # Filter extremely noisy text (too little alnum/hangul ratio)
    letters = len(re.findall(r"[A-Za-z0-9가-힣]", text))
    if letters < max(8, int(len(text) * 0.2)):
        return False
    return True


def split_into_units(text: str) -> List[str]:
    """Split by headings/bullets/paragraphs to keep semantics together."""
    t = normalize_text(text)
    # sentence-ish / bullet-ish boundaries → newline
    t = re.sub(r"([.!?]|\u2026)(?!\d)", r"\1\n", t)       # sentence enders
    t = re.sub(r"\n-\s+", "\n- ", t)                      # bullets
    # Now split on blank lines or bullets
    parts = re.split(r"(?:\n\s*\n|\n- )", t)
    # reattach leading '-' for bullet units we cut off
    fixed = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.startswith("- ") and re.match(r"^[-•●·▶►]\s+", p):
            p = "- " + re.sub(r"^[-•●·▶►]\s+", "", p)
        fixed.append(p)
    return fixed


def pack_units_to_chunks(units: List[str], tok, chunk_tokens: int, overlap_units: int = 1) -> List[Dict[str, Any]]:
    """Greedy pack units into token-limited chunks; keep small overlap in units."""
    chunks: List[Dict[str, Any]] = []
    if not units:
        return chunks

    token_cache = {}
    def ntoks(u: str) -> int:
        if u not in token_cache:
            token_cache[u] = len(tok(u, add_special_tokens=False, truncation=False)["input_ids"])
        return token_cache[u]

    cur: List[str] = []
    cur_tok = 0
    for u in units:
        u_tok = ntoks(u)
        if (cur and cur_tok + u_tok > chunk_tokens):
            chunks.append({"text": "\n".join(cur).strip()})
            # keep last overlap_units units for context continuity
            cur = cur[-overlap_units:] if overlap_units > 0 else []
            cur_tok = sum(ntoks(x) for x in cur)
        cur.append(u)
        cur_tok += u_tok

    if cur:
        chunks.append({"text": "\n".join(cur).strip()})

    return chunks


# ---------------------------
# Ingestor
# ---------------------------

class MilvusDBManager:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        collection_name: str = None,
        chunk_tokens: int = 256,
        chunk_overlap_units: int = 1,
        min_chunk_chars: int = 60,
        min_chunk_tokens: int = 0,
    ):
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.uri = os.getenv("MILVUS_URI")  # ex) 127.0.0.1:19530
        self.token = os.getenv("MILVUS_TOKEN")  # if auth enabled
        self.secure = os.getenv("MILVUS_SECURE", "false").lower() == "true"
        # 환경변수에 공백이 포함되는 경우가 있어 안전하게 trim
        self.collection_name = (collection_name or os.getenv("CONCEPT_COLL", "concepts")).strip()
        self.dimension = int(os.getenv("EMBED_DIM", "768"))
        self.chunk_tokens = int(os.getenv("CHUNK_TOKENS", chunk_tokens))
        self.chunk_overlap_units = int(os.getenv("CHUNK_OVERLAP_UNITS", chunk_overlap_units))
        self.min_chunk_chars = int(os.getenv("CHUNK_MIN_CHARS", min_chunk_chars))
        self.min_chunk_tokens = int(os.getenv("CHUNK_MIN_TOKENS", min_chunk_tokens))

        # embedding
        self.embeddings_model: SentenceTransformer | None = None
        self.collection: Collection | None = None

    # ---------- connections & schema ----------

    def connect(self) -> bool:
        try:
            # Prefer modern URI style if provided
            if self.uri:
                connections.connect(alias="default", uri=self.uri, token=self.token, secure=self.secure)
                log.info(f"Connected to Milvus via uri={self.uri}")
                return True
            # else build uri from host:port (more reliable than host/port in some client versions)
            uri = f"{self.host}:{self.port}"
            try:
                connections.connect(alias="default", uri=uri, token=self.token, secure=self.secure)
                log.info(f"Connected to Milvus via uri={uri}")
                return True
            except Exception as e2:
                log.warning(f"URI connect failed ({uri}), fallback to host/port: {e2}")
                connections.connect(alias="default", host=self.host, port=self.port)
                log.info(f"Connected to Milvus at {self.host}:{self.port}")
                return True
        except Exception as e:
            log.error(f"Milvus connect failed: {e}")
            return False

    def load_embedding_model(self) -> bool:
        try:
            model_name = os.getenv("EMBED_MODEL", "jhgan/ko-sroberta-multitask")
            self.embeddings_model = SentenceTransformer(model_name)
            test = self.embeddings_model.encode("임베딩 테스트")
            if len(test) != self.dimension:
                log.warning(f"Embedding dim {len(test)} != expected {self.dimension}. Updating to {len(test)}")
                self.dimension = len(test)
            log.info(f"Embedding model loaded: {model_name} (dim={self.dimension})")
            return True
        except Exception as e:
            log.error(f"Load embedding model failed: {e}")
            return False

    def _schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="item_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="item_title", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=16000),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="unit_count", dtype=DataType.INT64),
            FieldSchema(name="n_tokens", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        return CollectionSchema(fields=fields, description="RAG concepts (cleaned & chunked)")

    def create_collection(self, drop: bool = False) -> bool:
        if utility.has_collection(self.collection_name):
            if drop:
                utility.drop_collection(self.collection_name)
                log.info(f"Dropped existing collection '{self.collection_name}'")
            else:
                self.collection = Collection(self.collection_name)
                log.info(f"Loaded existing collection '{self.collection_name}'")
                return True
        self.collection = Collection(self.collection_name, self._schema())
        log.info(f"Created collection '{self.collection_name}'")
        return True

    def build_index(self) -> None:
        if self.collection is None:
            return
        # Choose index type
        index_type = os.getenv("MILVUS_INDEX", "HNSW").upper()  # HNSW recommended for cosine
        metric = os.getenv("MILVUS_METRIC", "COSINE").upper()
        if index_type == "HNSW":
            index_params = {"index_type": "HNSW", "metric_type": metric, "params": {"M": 16, "efConstruction": 200}}
        else:
            # IVF_FLAT as fallback
            n_entities = max(1, int(utility.num_entities(self.collection_name) or 1))
            nlist = max(64, min(65536, int(math.sqrt(n_entities)) * 4))
            index_params = {"index_type": "IVF_FLAT", "metric_type": metric, "params": {"nlist": nlist}}

        self.collection.create_index("embedding", index_params=index_params)
        self.collection.load()
        log.info(f"Built index ({index_type}/{metric}) and loaded collection.")

    # ---------- processing ----------

    @property
    def tok(self):
        if not self.embeddings_model:
            raise RuntimeError("Embedding model must be loaded before tokenization.")
        if not hasattr(self.embeddings_model, "tokenizer"):
            raise RuntimeError("SentenceTransformer tokenizer not available on this model.")
        return self.embeddings_model.tokenizer

    def _file_items(self, path: str) -> Tuple[str, List[Dict[str, Any]]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        subject = (data.get("subject") or "정보처리기사").strip()
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            # support single doc shape
            items = [data]
        return subject, items

    def _units_for_item(self, title: str, content: str) -> List[str]:
        title = normalize_text(title or "")
        content = normalize_text(content or "")
        base = title.strip()
        # Prefer to keep title in the first unit to strengthen retrieval
        units = split_into_units(content) or ([content] if content else [])
        if base:
            if units:
                units[0] = f"{base}\n{units[0]}"
            else:
                units = [base]
        return units

    def process_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Return cleaned, chunked items ready for embedding & insert."""
        subject, items = self._file_items(file_path)
        out: List[Dict[str, Any]] = []

        for it in items:
            if not isinstance(it, dict):
                continue
            item_id = str(it.get("item_id", ""))
            title = it.get("item_title", "") or ""
            content = it.get("content", "") or ""

            units = self._units_for_item(title, content)

            # filter noisy/short units early
            units = [u for u in units if good_enough(u, self.min_chunk_chars, self.min_chunk_tokens, self.tok)]
            if not units:
                continue

            # pack into chunks
            chunks = pack_units_to_chunks(units, self.tok, self.chunk_tokens, self.chunk_overlap_units)

            for cidx, ch in enumerate(chunks):
                text = ch["text"].strip()
                if not good_enough(text, self.min_chunk_chars, self.min_chunk_tokens, self.tok):
                    continue
                record = {
                    "subject": subject,
                    "source_file": os.path.basename(file_path),
                    "item_id": item_id if len(chunks) == 1 else f"{item_id}_chunk_{cidx+1}",
                    "item_title": title[:2000],
                    "content": text[:16000],
                    "content_hash": sha1(text),
                    "chunk_index": cidx,
                    "unit_count": len(units),
                    "n_tokens": len(self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]),
                }
                # deterministic id: uuid5 over (source_file + item_id + chunk_index + hash)
                ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
                record["id"] = str(uuid.uuid5(ns, f"{record['source_file']}|{record['item_id']}|{record['content_hash']}"))
                out.append(record)
        log.info(f"Processed {len(out)} chunks from '{os.path.basename(file_path)}'")
        return out

    def process_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process newline-delimited JSON, mapping title->subject, text->content."""
        out: List[Dict[str, Any]] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception as e:
            log.error(f"Read jsonl failed: {file_path} | {e}")
            return out

        for ln in lines:
            try:
                it = json.loads(ln)
            except Exception:
                continue

            # Map jsonl schema → internal schema
            subject = (it.get("title") or os.path.splitext(os.path.basename(file_path))[0]).strip()
            item_title = (it.get("metadata", {}).get("section_title") or subject or "").strip()[:2000]
            content = (it.get("text") or "").strip()
            if not content:
                continue

            # Prefer original source path if present in jsonl
            source_hint = it.get("source")
            source_file = os.path.basename(source_hint) if source_hint else os.path.basename(file_path)

            base_item_id = str(it.get("id") or f"{os.path.basename(file_path)}:{it.get('chunk_index', 0)}")

            units = self._units_for_item(item_title, content)
            units = [u for u in units if good_enough(u, self.min_chunk_chars, self.min_chunk_tokens, self.tok)]
            if not units:
                continue

            chunks = pack_units_to_chunks(units, self.tok, self.chunk_tokens, self.chunk_overlap_units)
            for cidx, ch in enumerate(chunks):
                text = ch["text"].strip()
                if not good_enough(text, self.min_chunk_chars, self.min_chunk_tokens, self.tok):
                    continue
                record = {
                    "subject": subject,
                    "source_file": source_file,
                    "item_id": base_item_id if len(chunks) == 1 else f"{base_item_id}_chunk_{cidx+1}",
                    "item_title": item_title,
                    "content": text[:16000],
                    "content_hash": sha1(text),
                    "chunk_index": cidx,
                    "unit_count": len(units),
                    "n_tokens": len(self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]),
                }
                ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
                record["id"] = str(uuid.uuid5(ns, f"{record['source_file']}|{record['item_id']}|{record['content_hash']}"))
                out.append(record)

        log.info(f"Processed {len(out)} chunks from jsonl '{os.path.basename(file_path)}'")
        return out

    # ---------- dedup & insert ----------

    def _existing_hashes(self, hashes: List[str]) -> set:
        """Query milvus for existing content_hashes (batched)."""
        if not hashes:
            return set()
        found: set = set()
        B = 800   # batched length for 'in' expr
        for i in range(0, len(hashes), B):
            batch = hashes[i:i+B]
            # 문자열 안전하게 더블쿼트로 감싸고 내부 특수문자도 이스케이프
            quoted = ",".join(json.dumps(h) for h in batch)   # h가 str이든 뭐든 안전
            expr = f"content_hash in [{quoted}]"
            try:
                rows = self.collection.query(
                    expr,
                    output_fields=["content_hash"],
                    consistency_level="Eventually",
                )
                for r in rows or []:
                    val = r.get("content_hash")
                    if val is not None:
                        found.add(val)
            except Exception as e:
                log.warning(f"query existing hashes failed (continuing): {e}")
                continue
        return found

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with normalization."""
        bs = int(os.getenv("EMBED_BATCH", "64"))
        return self.embeddings_model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

    def insert_records(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        # 1) dedup (in-file) by hash
        uniq = {}
        for r in records:
            uniq[r["content_hash"]] = r
        records = list(uniq.values())

        # 2) dedup (in-DB) by content_hash
        existing = self._existing_hashes([r["content_hash"] for r in records])
        records = [r for r in records if r["content_hash"] not in existing]
        if not records:
            log.info("No new records to insert (all duplicates).")
            return 0

        # 3) embeddings
        texts = [f"{r['item_title']} {r['content']}" for r in records]
        embeds = self.embed_texts(texts)

        # 4) insert
        insert_cols = [
            [r["id"] for r in records],
            [r["subject"] for r in records],
            [r["source_file"] for r in records],
            [r["item_id"] for r in records],
            [r["item_title"] for r in records],
            [r["content"] for r in records],
            [r["content_hash"] for r in records],
            [r["chunk_index"] for r in records],
            [r["unit_count"] for r in records],
            [r["n_tokens"] for r in records],
            embeds,
        ]
        self.collection.insert(insert_cols)
        self.collection.flush()
        log.info(f"Inserted {len(records)} records.")
        return len(records)

    # ---------- end-to-end ----------

    def load_all_json_files(self, json_dir: str) -> int:
        if not os.path.exists(json_dir):
            log.error(f"No such directory: {json_dir}")
            return 0
        files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json") or f.endswith(".jsonl")]
        files.sort()
        log.info(f"Found {len(files)} json/jsonl files under {json_dir}")

        total = 0
        for fp in files:
            if fp.lower().endswith(".jsonl"):
                recs = self.process_jsonl_file(fp)
            else:
                recs = self.process_json_file(fp)
            total += self.insert_records(recs)
        log.info(f"All done. Newly inserted: {total}")
        return total

    # ---------- search ----------
    def search_similar(self, query: str, top_k: int = 5, subject_filter: str | None = None) -> List[Dict[str, Any]]:
        if self.collection is None:
            log.error("Collection not loaded")
            return []
        try:
            self.collection.load()
        except Exception:
            pass

        qvec = self.embed_texts([query])[0]
        params = {"metric_type": os.getenv("MILVUS_METRIC", "COSINE"), "params": {}}
        if os.getenv("MILVUS_INDEX", "HNSW").upper() == "HNSW":
            params["params"]["ef"] = int(os.getenv("MILVUS_EF", "128"))
        else:
            params["params"]["nprobe"] = int(os.getenv("NPROBE", "32"))

        expr = None
        if subject_filter:
            expr = f'subject == "{subject_filter}"'

        res = self.collection.search(
            [qvec],
            "embedding",
            params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "subject", "source_file", "item_id", "item_title", "content", "chunk_index", "n_tokens"]
        )
        out = []
        for hits in res:
            for h in hits:
                out.append({
                    "id": h.entity.get("id"),
                    "score": float(h.score),
                    "subject": h.entity.get("subject"),
                    "source_file": h.entity.get("source_file"),
                    "item_id": h.entity.get("item_id"),
                    "item_title": h.entity.get("item_title"),
                    "content": h.entity.get("content"),
                    "chunk_index": h.entity.get("chunk_index"),
                    "n_tokens": h.entity.get("n_tokens"),
                })
        return out


def main():
    db = MilvusDBManager()
    if not db.connect():
        return
    if not db.load_embedding_model():
        return

    drop = os.getenv("MILVUS_DROP_COLLECTION", "false").lower() == "true"
    if not db.create_collection(drop=drop):
        return

    # ingest
    json_dir = os.getenv("JSON_DIR", "teacher/agents/retrieve/data/json")
    db.load_all_json_files(json_dir)

    # index (build after inserts)
    db.build_index()

    # quick smoke test
    q = os.getenv("TEST_QUERY", "데이터베이스 논리적 설계")
    res = db.search_similar(q, top_k=3)
    for i, r in enumerate(res, 1):
        log.info(f"{i}. score={r['score']:.4f} | {r['subject']} | {r['item_title'][:60]}")
        log.info(f"   {r['content'][:120]}...")


if __name__ == "__main__":
    main()
