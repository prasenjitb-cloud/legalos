"""Search pinned statutory snapshots by section and source unit."""

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import snowballstemmer

ACT_ID = "posh_2013"
EMBEDDING_MODEL = "BAAI/bge-small-en"
EMBEDDING_REVISION = "2275a7bdee235e9b4f01fa73aa60d3311983cfea"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
STOP_WORDS = ENGLISH_STOP_WORDS - {"against", "not", "no", "without"}
STEMMER = snowballstemmer.stemmer("english")
REVIEWED_PDF_SHA256 = "e59776d9ce4300c35acb8f3ed8150b33c18a77b5b553423ef9f1c69dbd144475"
SOURCE_SNAPSHOTS = {
    ACT_ID: (REVIEWED_PDF_SHA256, 30),
    "hma_1955": ("dbdc6d2a4223481908817b140e9c33d3bab6a48ff29787a7bd80786560171103", 37),
    "mva_1988": ("be5bac906607cb2ac902d5c113c42c7f89142be3591a81ccb825cfbb7ee5d50d", 256),
    "jja_2015": ("00d43505b7f4f254a657dcdb0c330b3dd13d093d40a3eb5074baa1be1283832c", 112),
    "cgst_2017": ("7c5f25425c67244e1621a3e142602350e70309a24c68833d475bf7a3c997725f", 188),
    "cpa_2019": ("414fb1edd9b0f7fbc2839ad92fb2b29ebfa3b557e70e928ab4a535802199e3e7", 107),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(text):
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS]
    return STEMMER.stemWords(words)


def section_order(section):
    match = re.fullmatch(r"(\d+)([A-Z]*)", section)
    if not match:
        raise ValueError("Invalid numbered section identifier: " + section)
    return int(match.group(1)), match.group(2)


def section_references(text):
    return [value.upper() for value in re.findall(r"\bsections?\s+(\d+[A-Z]?)\b", text, re.I)]


class EvidenceStore:
    def __init__(self, path, cache_dir, *, dense=True):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        self.act_id, self.act_name = self.data["act_id"], self.data["act_name"]
        expected = SOURCE_SNAPSHOTS.get(self.act_id)
        if not expected or self.data["source_sha256"] != expected[0]:
            raise ValueError("Unrecognised source snapshot")
        if any(p["act_id"] != self.act_id for p in self.data["provisions"]):
            raise ValueError("Mixed or unsupported Acts in the source store")
        self.provisions = {p["id"]: p for p in self.data["provisions"]}
        if len(self.provisions) != expected[1] or len(self.provisions) != len(self.data["provisions"]):
            raise ValueError("Incomplete or duplicate numbered sections")
        if self.act_id == ACT_ID and (self.data["source_updated"] != "2021-08-31" or
                                    set(self.provisions) != {str(i) for i in range(1, 31)}):
            raise ValueError("Expected the reviewed thirty-section POSH snapshot")
        self.units = {u["id"]: {**u, "section_id": p["id"], "act_id": p["act_id"]}
                      for p in self.provisions.values() for u in p["units"]}
        if len(self.units) != sum(len(p["units"]) for p in self.provisions.values()):
            raise ValueError("Duplicate evidence-unit identifiers")
        self.chunks = []
        self.rerank_passages = []
        for p in self.provisions.values():
            words = p["text"].split()
            for start in range(0, len(words), 140):
                self.chunks.append({"section_id": p["id"],
                                    "text": p["title"] + "\n" + " ".join(words[start:start + 180])})
            for unit in p["units"]:
                words = unit["text"].split()
                for start in range(0, len(words), 140):
                    self.rerank_passages.append({"section_id": p["id"], "unit_id": unit["id"],
                        "text": p["title"] + "\n" + " ".join(words[start:start + 180])})
        self.counts = [Counter(tokens(c["text"])) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(self.lengths)
        self.df = Counter(word for counts in self.counts for word in counts)
        self.encoder, self.vectors, self.reranker = None, None, None
        if dense:
            from langchain_huggingface import HuggingFaceEmbeddings
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
            from transformers import AutoTokenizer
            self.encoder = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu", "revision": EMBEDDING_REVISION},
                encode_kwargs={"normalize_embeddings": True},
            )
            cache = Path(cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            versions = ":".join(importlib.metadata.version(p) for p in ["torch", "transformers", "sentence-transformers"])
            key = hashlib.sha256((digest(path) + EMBEDDING_MODEL + EMBEDDING_REVISION + versions + "180:140:cpu:v1").encode()).hexdigest()
            target = cache / f"{key}.npy"
            if target.exists():
                self.vectors = np.load(target)
            else:
                self.vectors = np.array(self.encoder.embed_documents([c["text"] for c in self.chunks]))
                np.save(target, self.vectors)
            if not np.isfinite(self.vectors).all():
                raise RuntimeError("Embedding index contains non-finite values")
            graph = hf_hub_download(RERANKER_MODEL, "onnx/model.onnx", revision=RERANKER_REVISION)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.reranker = ort.InferenceSession(graph, sess_options=options, providers=["CPUExecutionProvider"])
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, revision=RERANKER_REVISION)

    def rerank(self, query, indices):
        scores = []
        names = {item.name for item in self.reranker.get_inputs()}
        for start in range(0, len(indices), 8):
            texts = [self.rerank_passages[i]["text"] for i in indices[start:start + 8]]
            encoded = self.rerank_tokenizer([query] * len(texts), texts, padding=True,
                                            truncation=True, max_length=512, return_tensors="np")
            feed = {name: value for name, value in encoded.items() if name in names}
            scores.extend(self.reranker.run(None, feed)[0].reshape(-1).tolist())
        if not np.isfinite(scores).all():
            raise RuntimeError("Reranker returned non-finite scores")
        return scores

    def fetch(self, act_id, section_id):
        if act_id != self.act_id or str(section_id) not in self.provisions:
            raise ValueError("Provision not found in the selected Act snapshot")
        return self.provisions[str(section_id)]

    def search(self, question_part, evidence_target, *, act_id=None, mode="hybrid", k=3, exclude=()):
        act_id = act_id or self.act_id
        if act_id != self.act_id or mode not in {"hybrid", "lexical", "semantic"}:
            raise ValueError("Invalid search scope or mode")
        query = question_part + " " + evidence_target
        scores = []
        for count, length in zip(self.counts, self.lengths):
            score = 0.0
            for word in set(tokens(query)):
                frequency = count[word]
                idf = math.log(1 + (len(self.chunks) - self.df[word] + 0.5) / (self.df[word] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / self.average))
            scores.append(score)
        rankings = {}
        if mode in {"hybrid", "lexical"}:
            rankings["bm25"] = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        if mode in {"hybrid", "semantic"}:
            if self.encoder is None:
                raise ValueError("Semantic index not loaded")
            dense_scores = self.vectors @ np.array(self.encoder.embed_query(QUERY_INSTRUCTION + query))
            if not np.isfinite(dense_scores).all():
                raise RuntimeError("Embedding search returned non-finite scores")
            rankings["semantic"] = sorted(range(len(scores)), key=lambda i: (-dense_scores[i], i))
        fused, methods = Counter(), {}
        for method, ranking in rankings.items():
            seen = set()
            for index in ranking:
                section = self.chunks[index]["section_id"]
                if section in seen or section in exclude:
                    continue
                if method == "bm25" and scores[index] <= 0:
                    continue
                seen.add(section)
                fused[section] += 1 / (60 + len(seen))
                methods.setdefault(section, {})[method] = len(seen)
        order = sorted(fused, key=lambda s: (-fused[s], section_order(s)))
        rerank_scores, chunk_rerank, rerank_ranks = {}, {}, {}
        if self.reranker and order:
            candidates = set(order[:12])
            indices = [i for i, c in enumerate(self.rerank_passages) if c["section_id"] in candidates]
            for name, rerank_query in [("question", question_part), ("target", evidence_target)]:
                values = self.rerank(rerank_query, indices)
                grouped = {}
                for index, value in zip(indices, values):
                    section = self.rerank_passages[index]["section_id"]
                    chunk_rerank.setdefault(index, {})[name] = float(value)
                    grouped[section] = max(grouped.get(section, float("-inf")), float(value))
                ranked = sorted(candidates, key=lambda s: (-grouped[s], section_order(s)))
                for rank, section in enumerate(ranked, 1):
                    rerank_ranks.setdefault(section, {})[name] = rank
                    rerank_scores[section] = rerank_scores.get(section, 0) + 1 / (60 + rank)
            question_first = next(s for s in candidates if rerank_ranks[s]["question"] == 1)
            target_first = next(s for s in candidates if rerank_ranks[s]["target"] == 1)
            for section in candidates:
                if "bm25" in methods[section]:
                    rerank_scores[section] += 1 / (60 + methods[section]["bm25"])
            leaders = sorted({question_first, target_first}, key=lambda s: (-rerank_scores[s], section_order(s)))
            order = leaders + sorted(candidates - set(leaders), key=lambda s: (-rerank_scores[s], section_order(s))) + [s for s in order if s not in candidates]
        self.last_search = {
            "query": query, "mode": mode, "act_id": act_id,
            "lexical_terms": sorted(set(tokens(query))), "embedding_query_instruction": QUERY_INSTRUCTION,
            "reranker": RERANKER_MODEL if self.reranker else None,
            "reranker_revision": RERANKER_REVISION if self.reranker else None,
            "reranker_backend": "onnxruntime_cpu" if self.reranker else None,
            "reranker_queries": {"question": question_part, "target": evidence_target},
            "chunk_scores": [{"chunk": i, "section_id": c["section_id"], "bm25": scores[i],
                              "semantic": float(dense_scores[i]) if "semantic" in rankings else None}
                             for i, c in enumerate(self.chunks)],
            "rerank_passages": [{**self.rerank_passages[i], "scores": value} for i, value in chunk_rerank.items()],
            "provision_ranks": [{"section_id": s, "score": fused[s], "ranks": methods[s],
                                 "rerank_score": rerank_scores.get(s), "rerank_ranks": rerank_ranks.get(s, {})} for s in order],
        }
        return [{"section_id": s, "score": fused[s], "ranks": methods[s],
                 "rerank_score": rerank_scores.get(s), "rerank_ranks": rerank_ranks.get(s, {})} for s in order[:k]]
