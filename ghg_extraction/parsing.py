from dataclasses import dataclass
from typing import Optional, Dict, Any

from abc import ABC, abstractmethod

from pathlib import Path
import json, os, re, time, torch, logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# AcceleratorDevice: a fixed set of named choices for the hardware device (CPU, CUDA, AUTO).
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, # standard CPU pipeline
    TableFormerMode, # Option choice between FAST or ACCURATE mode
    TableStructureOptions, # detailed settings for table-parsing model # TODO keep tracking
    ThreadedPdfPipelineOptions, # GPU pipeline accelarator options (batch size)
)

# The pipeline class used when processing on GPU.
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import ContentLayer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling.document_converter import DocumentConverter, PdfFormatOption


@dataclass
class ParsedDocument:
    markdown: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None
    txt: Optional[str] = None
    document_metadata: Optional[Dict[str, Any]] = None # stores company report specific metadata like amount of pages, texts, tables, pictures


class BasePdfBackend(ABC):
    @abstractmethod
    def parse(self, pdf_path: str) -> ParsedDocument:
        pass


class DoclingBackend(BasePdfBackend):
    def __init__(self, do_ocr: bool = False, use_vlm: bool = False):
        self._configure(do_ocr, use_vlm)

    def _configure(self, do_ocr: bool, use_vlm: bool):
        cuda_available = torch.cuda.is_available()

        
        self.pipeline_options = ThreadedPdfPipelineOptions(
            do_ocr=do_ocr,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.AUTO),
            ocr_batch_size=64 if cuda_available else 4,     # 64 regions processed by OCR in parallel
            layout_batch_size=64 if cuda_available else 4,  # 64 regions processed by layout model in parallel
            table_batch_size=4,    # TableFormer does not benefit from large batches
        )
        #self.pipeline_cls = ThreadedStandardPdfPipeline

        # ThreadedStandardPdfPipeline == StandardPdfPipeline (backwards-compat alias)
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=self.pipeline_options
            )}
        )

        # Table settings
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=False,
        )

        # TODO: VLM picture annotation — GPU only
        if use_vlm and cuda_available:
            # Tell docling to run picture description on all images found in the PDF.
            self.pipeline_options.do_picture_description = True
            self.pipeline_options.images_scale = 2.0
            self.pipeline_options.generate_page_images = True

            # TODO: configure the VLM model here when GPU hardware is available.
        else:
            self.pipeline_options.do_picture_description = False
            

        start_time = time.time()
        self.converter.initialize_pipeline(InputFormat.PDF)
        init_runtime = time.time() - start_time
        print(f"Pipeline initialized in {init_runtime:.2f} seconds.")


    def parse(self, pdf_path: str) -> ParsedDocument:
        start_time = time.time()
        conversion_result = self.converter.convert(pdf_path)
        pipeline_runtime = time.time() - start_time
        print(f"Document converted in {pipeline_runtime:.2f} seconds.")

        markdown = MarkdownDocSerializer(doc=conversion_result.document,
                                         params=MarkdownParams(
                                             layers={ContentLayer.BODY, ContentLayer.FURNITURE},
                                             include_hyperlinks=False,
                                             traverse_pictures=False, # True for text content attached as children under picture nodes - Only for OCR
                                             image_placeholder=" ",
                                             ),
                                         ).serialize().text

        return ParsedDocument(
            markdown=markdown,
            json_data=conversion_result.document.export_to_dict(),
            document_metadata={
                "backend": "docling",
                "conversion_time": pipeline_runtime,
                "pages": len(getattr(conversion_result.document, "pages", [])),
                "confidence_grade": conversion_result.confidence.mean_grade,
                "confidence_score": conversion_result.confidence.layout_score,
                "word_count": len(markdown.split()),
                "detected_tables": len(getattr(conversion_result.document, "tables", [])),
                "detected_pictures": len(getattr(conversion_result.document, "pictures", [])),
                }
            )

# TODO
class PyMuPdfBackend(BasePdfBackend):
    pass

class MarkerBackend(BasePdfBackend):
    pass

class PdfParser:
    """
    Parses PDF documents using the selected backend and saves the extracted outputs (e.g., Markdown, JSON, and document metadata) to disk.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the input and output directories.
    backend : str
        Name of the parsing backend to use. Must correspond to a registered
        backend (currently only "docling" is supported).
    **backend_kwargs
        Additional keyword arguments passed to the backend constructor.
        For example, the Docling backend supports:
            - do_ocr (bool)
            - use_vlm (bool)
    """

    BACKENDS = {
        "docling": DoclingBackend,
        "pymupdf": PyMuPdfBackend, # NOT IMPLEMENTED
        "marker": MarkerBackend, # NOT IMPLEMENTED
    }

    def __init__(self, config, backend: str, **backend_kwargs):
        self._pdf_dir = config.get("pdf_dir", config.get("test_pdf_dir", os.makedirs("data/test_data", exist_ok=True))) # Default to test_pdf_dir if pdf_dir is not provided
        self._json_dir = config["json_dir"]
        self._md_dir = config["md_dir"]
        self._document_metadata_dir = config["document_metadata_dir"]
        
        log_dir = config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._timestamp = timestamp

        log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

        self._logger = logging.getLogger(f"PdfParser_{timestamp}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = True # If True, shows logs in console
        self._logger.handlers.clear() # Clear any existing handlers to avoid duplicate logs

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)

        self._run_log = log_file

        try:
            backend_cls = self.BACKENDS[backend.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available: {', '.join(self.BACKENDS)}"
            )

        self.backend = backend_cls(**backend_kwargs)


    def _log_warning(self, message: str):
        self._logger.warning(message)
        
        
    def _make_metadata_record(self, company_name, report_name, pdf_path, status, error=None, document_metadata=None):
        record = {
            "company_name": company_name,
            "report_name": report_name,
            "pdf_path": pdf_path,
            "status": status,
            "error": "" if error is None else str(error),
        }
        if document_metadata:
            record.update(document_metadata)

        return record


    def _load_documents(self):
        """Loads the PDF documents from the PDF directory and stores them into a list of files."""

        if not os.path.isdir(self._pdf_dir):
            raise ValueError(f"PDF directory '{self._pdf_dir}' does not exist or is not a directory.")

        company_folders = sorted([f for f in os.listdir(self._pdf_dir) if os.path.isdir(os.path.join(self._pdf_dir, f))])

        if not company_folders:
            self._log_warning(f"No company folders found in PDF directory '{self._pdf_dir}'.")
            return []

        pdf_files = []

        for company_name in company_folders:
            company_path = os.path.join(self._pdf_dir, company_name)

            if not os.path.isdir(company_path):
                self._log_warning(f"Company folder '{company_path}' is not a directory.")
                continue

            company_pdfs  = [f for f in sorted(os.listdir(company_path)) if f.lower().endswith(".pdf")]

            if not company_pdfs:
                self._log_warning(f"No PDF files found in company folder '{company_path}'.")
                continue

            pattern = re.compile(rf"^{re.escape(company_name)}_\d{{4}}\.pdf$")

            for pdf_name in company_pdfs:
                if not pattern.fullmatch(pdf_name):
                    self._log_warning(f"PDF file '{pdf_name}' in company folder '{company_path}' does not match the expected naming convention '{company_name}_YYYY.pdf'.")
                    continue

                pdf_path = os.path.join(company_path, pdf_name)
                if not os.path.isfile(pdf_path):
                    self._log_warning(f"PDF file '{pdf_path}' is not a valid file.")
                    continue

                pdf_files.append(pdf_path)

        if not pdf_files:
            self._log_warning(f"No valid PDF files found in PDF directory '{self._pdf_dir}'.")
            return []

        self._logger.info(f"Found {len(pdf_files)} PDF(s) to process.")

        return pdf_files


    # Only for DoclingDocument indexing
    def _save_json(self, json_data, company_name, report_name):
        if json_data is None:
            return

        json_folder = Path(self._json_dir) / company_name
        json_folder.mkdir(parents=True, exist_ok=True)

        json_file = json_folder / f"{report_name}.json"

        with json_file.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps(json_data))


    def _save_markdown(self, markdown, company_name, report_name):
        if markdown is None:
            return

        md_folder = Path(self._md_dir) / company_name
        md_folder.mkdir(parents=True, exist_ok=True)

        md_file = md_folder / f"{report_name}.md"

        with md_file.open("w", encoding="utf-8") as fp:
            fp.write(markdown)


    def parse_documents(self, skip_existing: bool = False):
        """Parse documents and saves the results.

        Args:
            skip_existing (bool, optional): Whether to skip already existing documents. Defaults to False.
        """
        pdf_files = self._load_documents()

        successful = 0
        failed = 0
        warnings = 0
        metadata_records = []

        for pdf_file in tqdm(pdf_files, colour="yellow"):

            filename = os.path.basename(pdf_file)
            report_name = os.path.splitext(filename)[0]
            parent_folder = os.path.dirname(pdf_file)
            company_name = os.path.basename(parent_folder)

            self._logger.info(f"Processing: {filename} (company: {company_name})")
            
            if skip_existing:
                parsed_company_folder = os.path.join(self._json_dir, company_name)
                
                if os.path.isdir(parsed_company_folder):
                    self._logger.info(f"Already processed: {company_name}. Skipping...")
                    continue
                
            try:
                parsed_document = (self.backend.parse(pdf_file))
            except Exception as exc:
                failed += 1
                error = f"{type(exc).__name__}: {exc}"
                self._logger.exception(f"Parsing failed for: {filename} / company={company_name}")
                metadata_records.append(self._make_metadata_record(company_name, report_name, pdf_file, status="failed", error=error))
                continue
            
            document_metadata = parsed_document.document_metadata or {}

            if document_metadata.get("word_count", 0) < 100:
                warnings += 1
                self._log_warning(f"PLEASE CHECK MANUALLY: '{filename}' in {pdf_file} (is likely a scanned document)")
                metadata_records.append(self._make_metadata_record(company_name, report_name, pdf_file, status="warning", error="", document_metadata=document_metadata))
                continue

            self._save_json(parsed_document.json_data, company_name, report_name)

            self._save_markdown(parsed_document.markdown, company_name, report_name)
            
            successful += 1
            metadata_records.append(self._make_metadata_record(company_name, report_name, pdf_file, status="success", error="", document_metadata=document_metadata))
            
            os.makedirs(self._document_metadata_dir, exist_ok=True)
            metadata_summary_path = os.path.join(self._document_metadata_dir, f"summary_{self._timestamp}.csv")
            pd.DataFrame(metadata_records).to_csv(metadata_summary_path, index=False)
            
            
            