from pathlib import Path
import json, re, time, torch, logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    OcrAutoOptions, # picks the first available OCR engine (RapidOCR, EasyOCR, ...)
    PdfPipelineOptions,
    TableFormerMode, # Option choice between FAST or ACCURATE mode
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ContentLayer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams


MARKDOWN_PARAMS = MarkdownParams(
    layers={ContentLayer.BODY, ContentLayer.FURNITURE},
    include_hyperlinks=False,
    traverse_pictures=False, # True for text content attached as children under picture nodes - Only for OCR
    image_placeholder=" ",
)

MIN_WORD_COUNT = 100 # below this a report is flagged for manual check


class PdfParser:
    """
    Converts every report PDF in the PDF directory with docling.

    Parameters
    ----------
    config : dict
        Must contain "pdf_dir".
    do_ocr : bool
        OCR only runs on page regions without a PDF text layer (scanned pages) and on pages
        whose text layer is broken.

    Reports are always converted one at a time.
    """

    def __init__(self, config, do_ocr: bool = True):
        self._pdf_dir = Path(config["pdf_dir"])
        self._output_dir = self._pdf_dir.parent.parent / "output"
        self._converted_dir = self._output_dir / "converted"

        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._logger = self._make_logger(self._output_dir / "logs")

        cuda_available = torch.cuda.is_available()

        page_batch_size = 64 if cuda_available else 4 # pages of ONE report sent through a model at once
        pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.CUDA if cuda_available else AcceleratorDevice.CPU
            ),
            do_ocr=do_ocr,
            ocr_options=OcrAutoOptions(replace_broken_text_layer=True),
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE,
                do_cell_matching=False,
            ),
            ocr_batch_size=page_batch_size,
            layout_batch_size=page_batch_size,
            table_batch_size=4, # TableFormer does not benefit from large batches
        )

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        start_time = time.time()
        self.converter.initialize_pipeline(InputFormat.PDF)
        self._logger.info(
            f"Pipeline initialized in {time.time() - start_time:.2f} seconds "
            f"(cuda={cuda_available}, do_ocr={do_ocr})."
        )


    def _make_logger(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(f"PdfParser_{self._timestamp}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear() # Clear any existing handlers to avoid duplicate logs

        file_handler = logging.FileHandler(log_dir / f"pipeline_{self._timestamp}.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logger.addHandler(file_handler)

        return logger


    def _load_documents(self):
        """Returns all PDFs in PDF directory."""

        if not self._pdf_dir.is_dir():
            raise ValueError(f"PDF directory '{self._pdf_dir}' does not exist or is not a directory.")

        pdf_files = []

        for company_path in sorted(p for p in self._pdf_dir.iterdir() if p.is_dir()):
            company_pdfs = sorted(f for f in company_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf")

            if not company_pdfs:
                self._logger.warning(f"No PDF files found in company folder '{company_path}'.")
                continue

            pattern = re.compile(rf"{re.escape(company_path.name)}_\d{{4}}\.pdf", re.IGNORECASE)

            for pdf_path in company_pdfs:
                if not pattern.fullmatch(pdf_path.name):
                    self._logger.warning(f"'{pdf_path}' does not match the naming convention '{company_path.name}_YYYY.pdf'. Parsing anyway.")
                pdf_files.append(pdf_path)

        self._logger.info(f"Found {len(pdf_files)} PDF(s) in '{self._pdf_dir}'.")

        return pdf_files


    def _output_path(self, pdf_path: Path, suffix: str):
        return self._converted_dir / pdf_path.parent.name / f"{pdf_path.stem}{suffix}"


    def _make_metadata_record(self, pdf_path: Path, status, error=""):
        return {
            "company_name": pdf_path.parent.name,
            "report_name": pdf_path.stem,
            "pdf_path": str(pdf_path),
            "status": status,
            "error": error,
        }


    def _save_summary(self, metadata_records):
        summary_path = self._output_dir / f"summary_{self._timestamp}.csv"
        pd.DataFrame(metadata_records).to_csv(summary_path, index=False)


    def parse_documents(self, skip_existing: bool = True):
        """Parse documents and saves the results.

        Args:
            skip_existing (bool, optional): Skip PDFs whose markdown output already exists. Defaults to True.
        """
        pdf_files = self._load_documents()
        metadata_records = []

        if skip_existing:
            to_convert = []
            for pdf_path in pdf_files:
                if self._output_path(pdf_path, ".md").exists():
                    self._logger.info(f"Already converted: {pdf_path}. Skipping...")
                    metadata_records.append(self._make_metadata_record(pdf_path, status="skipped"))
                else:
                    to_convert.append(pdf_path)
            pdf_files = to_convert

        for pdf_path in tqdm(pdf_files, colour="yellow"):
            self._logger.info(f"Processing: {pdf_path}")
            try:
                start_time = time.time()
                # raises_on_error=False: a failing report returns a FAILURE result instead of raising
                conversion_result = self.converter.convert(pdf_path, raises_on_error=False)
                conversion_time = time.time() - start_time
                record = self._handle_result(conversion_result, pdf_path, conversion_time)
            except Exception as exc:
                self._logger.exception(f"Parsing failed for: {pdf_path}")
                record = self._make_metadata_record(pdf_path, status="failed", error=f"{type(exc).__name__}: {exc}")

            metadata_records.append(record)
            self._save_summary(metadata_records) # after every report, so a crash keeps all finished records

        if not pdf_files:
            self._save_summary(metadata_records)

        status_counts = pd.Series([r["status"] for r in metadata_records]).value_counts().to_dict()
        self._logger.info(f"Finished: {status_counts}")


    def _handle_result(self, conversion_result, pdf_path: Path, conversion_time: float):
        status = conversion_result.status
        error = "; ".join(e.error_message for e in conversion_result.errors)

        if status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            self._logger.error(f"Parsing failed for: {pdf_path} | {status.value} | {error}")
            return self._make_metadata_record(pdf_path, status="failed", error=error)

        document = conversion_result.document
        markdown = MarkdownDocSerializer(doc=document, params=MARKDOWN_PARAMS).serialize().text

        json_path = self._output_path(pdf_path, ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(document.export_to_dict()), encoding="utf-8")
        # Markdown is written last, so its existence marks a completed report (used by skip_existing)
        self._output_path(pdf_path, ".md").write_text(markdown, encoding="utf-8")

        word_count = len(markdown.split())

        if status == ConversionStatus.PARTIAL_SUCCESS:
            self._logger.warning(f"Partially converted: {pdf_path} | {error}")
            record_status = "partial_success"
        elif word_count < MIN_WORD_COUNT:
            self._logger.warning(f"PLEASE CHECK MANUALLY: {pdf_path} has only {word_count} words.")
            record_status = "warning"
        else:
            record_status = "success"

        confidence = conversion_result.confidence
        record = self._make_metadata_record(pdf_path, status=record_status, error=error)
        record.update({
            "conversion_time": conversion_time,
            "pages": conversion_result.input.page_count,
            "word_count": word_count,
            "detected_tables": len(document.tables),
            "detected_pictures": len(document.pictures),
            "mean_grade": confidence.mean_grade.value,
            "low_grade": confidence.low_grade.value,
            "parse_score": confidence.parse_score, # quality of the PDF text layer; NaN if there is none (scanned report)
            "layout_score": confidence.layout_score,
            "table_score": confidence.table_score,
            "ocr_score": confidence.ocr_score, # NaN if no region was OCRed
        })

        return record
