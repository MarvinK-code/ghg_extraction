# GHG Extraction

Extracts greenhouse gas (GHG) emission data from company reports (PDF).

| Stage     | Code                          | Input         | Output                                    |
| --------- | ----------------------------- | ------------- | ----------------------------------------- |
| Parsing   | `ghg_extraction/parsing.py`   | report PDFs   | Markdown and docling JSON, one per report |
| Indexing  | `ghg_extraction/indexing.py`  | docling JSON  | one FAISS vector store per report         |
| Retrieval | `ghg_extraction/retriever.py` | vector stores | top-k pages per report, as one JSON file  |

TBD.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

This creates the `.venv` folder and installs all packages from `uv.lock`.

## Data

Put the reports into `data/reports/` or configure your own data directory (see Configuration), with **one folder per company**. A company folder can hold reports of several years:

```
data/
├── reports/
│   ├── Company A/
│   │   └── Company A_2021.pdf
│   └── Company B/
│       ├── Company B_2019.pdf
│       └── Company B_2020.pdf
```

**NOTE**: File names **must follow** `<company name>_<year>.pdf`. The file name without `.pdf` is the `report_id` used in all later stages, for example `Company A_2021`.

## Configuration

Settings are in `config.yaml`, all prompts and queries are in `prompts.yaml`.

**`config.yaml`**

| Setting                 | Example                       | Meaning                                                                |
| ----------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| `pdf_dir`               | `"data/reports"`              | folder with the report PDFs                                            |
| `output_dir`            | `"output"`                    | output folder of indexing and retrieval                                |
| `embedding_model_name`  | `"Qwen/Qwen3-Embedding-0.6B"` | embedding model for indexing and dense retrieval                       |
| `retrieval_method`      | `"dense"`                     | `"dense"`, `"bm25"` or `"hybrid"`                                      |
| `top_k`                 | `5`                           | number of pages kept per report                                        |
| `use_instruction_query` | `true`                        | dense and hybrid: put `instruction_query` in front of the search query |
| `enable_reranking`      | `true`                        | rerank the retrieved pages with a cross-encoder                        |
| `reranker_model_name`   | `"Qwen/Qwen3-Reranker-0.6B"`  | cross-encoder, only loaded if `enable_reranking` is `true`             |

**NOTE**: Parsing does not read `output_dir`. It writes to the folder `output` two levels above `pdf_dir`, which is the project root for `data/reports`. Keep `output_dir` at `"output"`, otherwise indexing does not find the parsed reports.

**`prompts.yaml`**

| Prompt              | Used by                 | Meaning                                                          |
| ------------------- | ----------------------- | ---------------------------------------------------------------- |
| `search_query`      | dense, hybrid, reranker | the one global question, the same for all reports                |
| `instruction_query` | dense, hybrid, reranker | task instruction for the Qwen models                             |
| `bm25_query`        | BM25, hybrid            | keyword list; BM25 searches only this list, never `search_query` |

TBD.

## Run

```bash
uv run main.py
```

`main.py` runs the stages one after another. Comment out the stages you do not want to run. Currently, parsing and indexing are commented out and only the retrieval runs.

Parsing and indexing may take a long time, especially on a CPU. Reports that were already parsed or indexed are skipped, so you can stop and restart these stages at any time. The retrieval does not skip anything and overwrites existing result files.

## Output

### Parsing

Every report PDF is converted into Markdown and JSON with [docling](https://github.com/docling-project/docling).

The `output/` folder is created automatically in the project root. Before continuing with the actual extraction, it must be ensured that the documents are converted (parsed).

```
output/
├── converted/
│   └── Company A/
│       ├── Company A_2021.md       # text of the report as Markdown
│       └── Company A_2021.json     # full docling document (structure, tables, positions)
...
```

#### How parsing works

- Reports are converted one at a time. A broken PDF is marked `failed` and the run continues.
- **OCR is applied only where needed:** pages without a text layer (scanned pages) and pages with a broken text layer are read with OCR. Normal digital pages keep their original text.
- A GPU is used automatically if one is available (needs a CUDA build of PyTorch, see [docling documentation](https://docling-project.github.io/docling/usage/gpu/)). Otherwise everything runs on CPU.

The code is in `ghg_extraction/parsing.py`.

#### Docling pipeline configuration

- **OCR**: OCR is enabled and uses the first available engine; RapidOCR (torch) as default. If you want to use other OCR engines install them via `uv add "docling[easyocr]"`.

- **Tables**

| Setting                                    | Value      | Meaning                                                                                                     |
| ------------------------------------------ | ---------- | ----------------------------------------------------------------------------------------------------------- |
| `do_table_structure`                       | `True`     | rows, columns and cells of each table are recognised (TableFormer)                                          |
| `table_structure_options.mode`             | `ACCURATE` | slower but more precise than `FAST`                                                                         |
| `table_structure_options.do_cell_matching` | `True`     | predicted cells are matched to the PDF's text cells and cleaned up (`False` can produce huge broken tables) |

- **Pictures** (detected, but not processed)
  This project currently bypasses any graphical visualisations for processing in subsequent modules. Consequently, numbers that only appear inside a chart or image are **not included** in the output.

- **Markdown export**

| Setting              | Value                  | Meaning                                                                      |
| -------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| `layers`             | `BODY` and `FURNITURE` | main text **and** page headers/footers are included                          |
| `include_hyperlinks` | `False`                | links are written as plain text                                              |
| `traverse_pictures`  | `False`                | text that docling places inside a picture (e.g. chart labels) is not written |
| `image_placeholder`  | `" "` (a blank)        | each picture is replaced by a blank instead of `<!-- image -->`              |

- **JSON export:** the complete `DoclingDocument` (`export_to_dict()`), including body and furniture, tables with cells, picture regions, and the page and position of every element.

### Indexing

The code is in `ghg_extraction/indexing.py` (class `ReportIndexer`). It builds **one FAISS vector store per report, with one chunk per page**, using only the parsed JSON files. A company with several reports gets several vector stores.

Only the page text is embedded, with the same Markdown settings as in parsing: main text **and** page headers/footers, links as plain text, pictures replaced by a blank. Metadata is **not** embedded.

### Retrieval

The code is in `ghg_extraction/retriever.py` (class `RetrieverModule`). The retrieval runs **for every report individually**: there is one global search query, and each report gets its own top-k pages.

#### Dense retrieval (method `dense`)

`DenseRetriever` embeds the search query and returns the pages with the highest cosine similarity.

(Optional):
With `use_instruction_query: true`, the query is written in the format Qwen3-Embedding expects:

```
Instruct: <instruction_query>
Query: <search_query>
```

With `false`, only `search_query` is embedded. The pages themselves are always embedded without an instruction.

#### BM25 retrieval (method `bm25`)

`KeywordRetriever` is a keyword search on the page texts with llama-index's `BM25Retriever`. It always searches `bm25_query`.

#### Hybrid retrieval (method `hybrid`)

`HybridRetriever` runs dense and BM25 retrieval and combines both rankings with reciprocal rank fusion (RRF, `ghg_extraction/helper/fusion.py`). Pages are matched by their `chunk_id`.

`use_instruction_query` applies to the dense part only, if enabled (`use_instruction_query: true`).

#### Reranking (optional)

With `enable_reranking: true`, any of the three methods first retrieves `max(top_k, 20)` candidates. For hybrid, reranking runs once, after the fusion. The cross-encoder then reads the search query together with each candidate page, and the best `top_k` pages are kept.

**ATTENTION**

The parameter `batch_size` in `ghg_extraction/retriever.py` is set to 1. Changing this parameter will influence the output significantly. We recommend **not changing** this value and keep it set to 1 (default is 32 according to the [documentation](https://sbert.net/docs/package_reference/cross_encoder/model.html#sentence_transformers.cross_encoder.model.CrossEncoder.rank))

### Generation

TBD.
