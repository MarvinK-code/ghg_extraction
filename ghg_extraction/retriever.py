from dataclasses import dataclass, asdict, replace
from pathlib import Path
from datetime import datetime
import json, logging

import torch
import Stemmer
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever


N_CANDIDATES = 20 # pages retrieved before reranking
RRF_K = 60 # standard constant of reciprocal rank fusion


def get_detailed_instruct(task_description: str, search_query: str) -> str:
    """Query format of Qwen3-Embedding; the pages themselves were embedded without an instruction."""
    return f"Instruct: {task_description}\nQuery: {search_query}"


@dataclass
class RetrievedItem:
    chunk_id: str # llama-index node id, "<report_id>_p<page_no>"
    text: str
    score: float
    metadata: dict


@dataclass
class RetrievalOutput:
    report_id: str
    method: str
    instruction_tuned: bool
    top_k: int
    reranked: bool
    items: list[RetrievedItem]


def to_items(nodes):
    """Converts llama-index results (NodeWithScore) into RetrievedItems."""
    return [
        RetrievedItem(chunk_id=n.node.node_id, text=n.node.get_content(), score=float(n.score), metadata=n.node.metadata)
        for n in nodes
    ]


class DenseRetriever:
    """Embedding similarity; the search query is prefixed with the instruction if one is given."""
    def __init__(self, index, instruction_query: str | None):
        self.index = index
        self.instruction_query = instruction_query

    def retrieve(self, search_query: str, top_k: int):
        if self.instruction_query:
            search_query = get_detailed_instruct(self.instruction_query, search_query)
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return to_items(retriever.retrieve(search_query))


class KeywordRetriever:
    """BM25 on the page texts, always searched with the fixed keyword list (bm25_query)."""
    def __init__(self, index, bm25_query: str):
        nodes = list(index.docstore.docs.values())
        self.bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=len(nodes), stemmer=Stemmer.Stemmer("english"))
        self.bm25_query = bm25_query

    def retrieve(self, top_k: int):
        nodes = [n for n in self.bm25.retrieve(self.bm25_query) if n.score > 0] # score 0 = no keyword on the page
        return to_items(nodes[:top_k])


class HybridRetriever:
    """Dense and BM25 combined with reciprocal rank fusion (RRF)."""
    def __init__(self, index, instruction_query: str | None, bm25_query: str):
        self.dense_retriever = DenseRetriever(index, instruction_query)
        self.keyword_retriever = KeywordRetriever(index, bm25_query)

    def retrieve(self, search_query: str, top_k: int):
        dense_items = self.dense_retriever.retrieve(search_query, top_k)
        keyword_items = self.keyword_retriever.retrieve(top_k)

        # A page scores 1 / (RRF_K + rank) in each ranking it appears in; pages are matched by chunk_id
        fused = {}
        for items in (dense_items, keyword_items):
            for rank, item in enumerate(items, start=1):
                if item.chunk_id not in fused:
                    fused[item.chunk_id] = replace(item, score=0.0)
                fused[item.chunk_id].score += 1 / (RRF_K + rank)

        return sorted(fused.values(), key=lambda item: item.score, reverse=True)[:top_k]


class RetrieverModule:
    """
    Runs the retrieval for every report with one global search query and saves the results.

    Parameters
    ----------
    config : dict
        Settings from config.yaml. Must contain "output_dir", "embedding_model_name", "retrieval_method"
        ("dense", "bm25" or "hybrid") and "top_k". Optional: "use_instruction_query" (dense and hybrid only)
        and "enable_reranking" with "reranker_model_name".
    prompts : dict
        Prompts from prompts.yaml. Must contain "search_query", "bm25_query" and "instruction_query".
    """

    def __init__(self, config, prompts):
        self._output_dir = Path(config["output_dir"])
        self._vectorstore_dir = self._output_dir / "vectorstores"

        self.search_query = prompts["search_query"]
        self.bm25_query = " ".join(prompts["bm25_query"].split()) # keyword list may span several lines
        self.instruction_query = prompts["instruction_query"]
        self.method = config["retrieval_method"]
        self.top_k = config["top_k"]
        self.enable_reranking = config.get("enable_reranking", False)

        if self.method not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"Unknown retrieval method '{self.method}'. Use 'dense', 'bm25' or 'hybrid'.")

        # BM25 always searches the keyword list
        self.use_instruction_query = config.get("use_instruction_query", False) and self.method != "bm25"

        method_name = self.method
        if self.use_instruction_query:
            method_name += "_instruct"
        if self.enable_reranking:
            method_name += "_rerank"
        self._retrieval_dir = self._output_dir / "retrieval" / method_name

        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._logger = self._make_logger(self._output_dir / "logs")

        self._embedding_model_name = config["embedding_model_name"]
        self.embed_model = HuggingFaceEmbedding(model_name=self._embedding_model_name)
        self.cross_encoder = CrossEncoder(config["reranker_model_name"]) if self.enable_reranking else None


    def _make_logger(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(f"RetrieverModule_{self._timestamp}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear() # Clear any existing handlers to avoid duplicate logs

        file_handler = logging.FileHandler(log_dir / f"retrieval_{self._timestamp}.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logger.addHandler(file_handler)

        return logger


    def _load_index(self, report_dir: Path):
        """Loads the FAISS vector store of one report."""
        manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["embedding_model_name"] != self._embedding_model_name:
            raise ValueError(f"Indexed with '{manifest['embedding_model_name']}', not '{self._embedding_model_name}'.")

        vector_store = FaissVectorStore.from_persist_dir(str(report_dir))
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(report_dir))

        return load_index_from_storage(storage_context, embed_model=self.embed_model)


    def _retrieve(self, index, top_k: int):
        instruction_query = self.instruction_query if self.use_instruction_query else None
        if self.method == "dense":
            return DenseRetriever(index, instruction_query).retrieve(self.search_query, top_k)
        if self.method == "bm25":
            return KeywordRetriever(index, self.bm25_query).retrieve(top_k)
        return HybridRetriever(index, instruction_query, self.bm25_query).retrieve(self.search_query, top_k)


    def _rerank(self, items):
        """Re-scores the candidates with the cross-encoder against the search query and keeps the top_k."""
        ranks = self.cross_encoder.rank(
            self.search_query,
            [item.text for item in items],
            top_k=self.top_k,
            prompt=self.instruction_query, # Qwen3-Reranker always takes an instruction, separately from the query
            activation_fn=torch.nn.Sigmoid(), # score = probability that the page is relevant
        )
        return [replace(items[rank["corpus_id"]], score=float(rank["score"])) for rank in ranks]


    def run_retrieval(self):
        """Retrieves the top_k pages of every indexed report and saves them as JSON."""
        report_dirs = sorted(path.parent for path in self._vectorstore_dir.glob("*/*/manifest.json"))
        self._logger.info(f"Found {len(report_dirs)} indexed report(s). Method: {self._retrieval_dir.name}, top_k={self.top_k}.")

        n_success = n_failed = 0
        for report_dir in tqdm(report_dirs, colour="blue"):
            company, report_id = report_dir.parent.name, report_dir.name
            output_path = self._retrieval_dir / company / f"{report_id}.json"

            try:
                index = self._load_index(report_dir)
                if self.enable_reranking:
                    items = self._rerank(self._retrieve(index, max(self.top_k, N_CANDIDATES)))
                else:
                    items = self._retrieve(index, self.top_k)

                output = RetrievalOutput(
                    report_id=report_id,
                    method=self.method,
                    instruction_tuned=self.use_instruction_query,
                    top_k=self.top_k,
                    reranked=self.enable_reranking,
                    items=items,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(asdict(output), indent=2, ensure_ascii=False), encoding="utf-8")

                self._logger.info(f"Retrieved: {report_id} -> pages {[item.metadata['page_no'] for item in items]}")
                n_success += 1
            except Exception:
                self._logger.exception(f"Retrieval failed for: {report_id}")
                n_failed += 1

        self._logger.info(f"Finished: success={n_success}, failed={n_failed}")
