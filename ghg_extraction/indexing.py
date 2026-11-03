from pathlib import Path
import json, re, shutil, time, logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

import faiss
from transformers import AutoTokenizer
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from docling_core.types.doc.document import ContentLayer, DoclingDocument
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider
from docling_core.transforms.chunker.page_chunker import PageChunker


# Same Markdown settings as parsing, but only the BODY layer: page headers/footers repeat on
# every page and would blur the page embeddings.
MARKDOWN_PARAMS = MarkdownParams(
    layers={ContentLayer.BODY},
    include_hyperlinks=False,
    traverse_pictures=False,
    image_placeholder=" ",
)


class MarkdownSerializerProvider(ChunkingSerializerProvider):
    """Makes the PageChunker write each page as Markdown with MARKDOWN_PARAMS."""
    def get_serializer(self, doc: DoclingDocument):
        return MarkdownDocSerializer(doc=doc, params=MARKDOWN_PARAMS)


class ReportIndexer:
    """
    Builds one FAISS vector store per parsed report, with one chunk per page.

    Parameters
    ----------
    config : dict
        Must contain "output_dir" and "embedding_model_name".
    """

    def __init__(self, config):
        self._output_dir = Path(config["output_dir"])
        self._converted_dir = self._output_dir / "converted"
        self._vectorstore_dir = self._output_dir / "vectorstores"

        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._logger = self._make_logger(self._output_dir / "logs")

        self._embedding_model_name = config["embedding_model_name"]
        self.embed_model = HuggingFaceEmbedding(model_name=self._embedding_model_name)
        self._embed_dim = self.embed_model._model.get_embedding_dimension()
        self._tokenizer = AutoTokenizer.from_pretrained(self._embedding_model_name)
        self._max_tokens = self.embed_model.max_length

        self._logger.info(
            f"Embedding model '{self._embedding_model_name}' loaded "
            f"(dim={self._embed_dim}, max_tokens={self._max_tokens})."
        )


    def _make_logger(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(f"ReportIndexer_{self._timestamp}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear() # Clear any existing handlers to avoid duplicate logs

        file_handler = logging.FileHandler(log_dir / f"indexing_{self._timestamp}.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logger.addHandler(file_handler)

        return logger


    def _load_documents(self):
        """Returns all parsed report JSONs in output_dir/converted/...."""

        if not self._converted_dir.is_dir():
            raise ValueError(f"Converted directory '{self._converted_dir}' does not exist. Run the parsing first.")

        json_files = sorted(self._converted_dir.glob("*/*.json"))
        self._logger.info(f"Found {len(json_files)} parsed report(s) in '{self._converted_dir}'.")

        return json_files


    def _report_dir(self, json_path: Path):
        return self._vectorstore_dir / json_path.parent.name / json_path.stem


    def _is_indexed(self, report_dir: Path):
        """A report counts as indexed if its manifest exists and was built with the current embedding model."""
        manifest_path = report_dir / "manifest.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("embedding_model_name") == self._embedding_model_name


    def _make_nodes(self, doc: DoclingDocument, json_path: Path):
        report_id = json_path.stem

        # One chunk per page; empty pages (e.g. blank or picture-only) are skipped by the chunker.
        # A paragraph that continues on the next page is one item in docling and belongs to its first page.
        nodes = []
        for chunk in PageChunker(serializer_provider=MarkdownSerializerProvider()).chunk(doc):
            text = chunk.text.strip()
            
            # the page number is determined by the first element (doc_items[0]) that occurs in the chunk (though the page number could be determined by the number of chunks since each chunk reflects a page)
            page_no = chunk.meta.doc_items[0].prov[0].page_no

            metadata = {
                "report_id": report_id,
                "page_no": page_no,
                "n_pages": len(doc.pages),
                "word_count": len(text.split()),
                "n_tokens": len(self._tokenizer(text)["input_ids"]),
                "source_json": str(json_path),
            }

            nodes.append(TextNode(
                id_=f"{report_id}_p{page_no}",
                text=text,
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata), # embed the page text only
            ))

        return nodes


    def _make_record(self, json_path: Path, status, error=""):
        return {
            "report_id": json_path.stem,
            "source_json": str(json_path),
            "status": status,
            "error": error,
        }


    def _save_summary(self, records):
        summary_path = self._output_dir / f"indexing_summary_{self._timestamp}.csv"
        pd.DataFrame(records).to_csv(summary_path, index=False)


    def create_vector_stores(self, skip_existing: bool = True):
        """Builds one vector store per report.

        Args:
            skip_existing (bool, optional): Skip reports already indexed with the current embedding model. Defaults to True.
        """
        json_files = self._load_documents()
        records = []

        for json_path in tqdm(json_files, colour="green"):
            report_dir = self._report_dir(json_path)

            if skip_existing and self._is_indexed(report_dir):
                self._logger.info(f"Already indexed: {json_path}. Skipping...")
                records.append(self._make_record(json_path, status="skipped"))
                continue

            self._logger.info(f"Indexing: {json_path}")
            try:
                record = self._index_report(json_path, report_dir)
            except Exception as exc:
                self._logger.exception(f"Indexing failed for: {json_path}")
                record = self._make_record(json_path, status="failed", error=f"{type(exc).__name__}: {exc}")

            records.append(record)
            self._save_summary(records) # after every report, so a crash keeps all finished records

        if not json_files:
            self._save_summary(records)

        status_counts = pd.Series([r["status"] for r in records]).value_counts().to_dict()
        self._logger.info(f"Finished: {status_counts}")


    def _index_report(self, json_path: Path, report_dir: Path):
        start_time = time.time()

        doc = DoclingDocument.load_from_json(json_path)
        nodes = self._make_nodes(doc, json_path)
        if not nodes:
            self._logger.warning(f"No page with text in: {json_path}")
            return self._make_record(json_path, status="failed", error="no page with text")

        max_tokens = max(node.metadata["n_tokens"] for node in nodes)
        n_truncated = sum(node.metadata["n_tokens"] > self._max_tokens for node in nodes)
        if n_truncated:
            self._logger.warning(f"{n_truncated} page(s) of {json_path} exceed {self._max_tokens} tokens and are truncated.")

        # Rebuild from scratch, so no stale files of an older index remain
        if report_dir.exists():
            shutil.rmtree(report_dir)
        report_dir.mkdir(parents=True)

        # Inner product on normalized embeddings (HuggingFaceEmbedding default) = cosine similarity
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(self._embed_dim))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=self.embed_model, show_progress=True)
        index.storage_context.persist(persist_dir=str(report_dir))

        indexing_time = time.time() - start_time

        # The manifest is written last, so its existence marks a complete vector store (used by skip_existing)
        manifest = {
            "report_id": json_path.stem,
            "source_json": str(json_path),
            "embedding_model_name": self._embedding_model_name,
            "embed_dim": self._embed_dim,
            "n_pages": len(doc.pages),
            "n_indexed_pages": len(nodes), # this might differ
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        record = self._make_record(json_path, status="success")
        record.update({
            "indexing_time": indexing_time,
            "n_pages": len(doc.pages),
            "n_indexed_pages": len(nodes),
            "n_empty_pages": len(doc.pages) - len(nodes),
            "max_tokens": max_tokens,
            "n_truncated_pages": n_truncated,
            "vectorstore_dir": str(report_dir),
        })

        return record
