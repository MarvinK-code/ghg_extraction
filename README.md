# GHG Extraction

Extracts greenhouse gas (GHG) emission data from company reports (PDF).

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

**NOTE**: File names **must follow** `<company name>_<year>.pdf`.

## Configuration

`config.yaml` only needs the path to the reports:

```yaml
pdf_dir: "data/reports"
```

## Run

```bash
uv run main.py
```

Parsing all reports takes a long time, especially on a CPU. Reports that were already converted are skipped, so you can stop and restart the run at any time.

## Output

### Parsing

Every report PDF is converted into Markdown and JSON with [docling](https://github.com/docling-project/docling).

The `output/` folder is created automatically in the project root (not stored in git). Before continue with the actual extraction, it must be ensured that the documents are converted (parsed).

```
output/
├── converted/
│   └── Company A/
│       ├── Company A_2021.md       # text of the report as Markdown
│       └── Company A_2021.json     # full docling document (structure, tables, positions)
├── logs/
│   └── pipeline_<timestamp>.log    # one log file per run
└── summary_<timestamp>.csv         # one row per report, one file per run
```

The summary CSV has one row per report with, among others:

| Column                                 | Meaning                                                                                               |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `status`                               | `success`, `partial_success`, `warning` (fewer than 100 words, check manually), `failed` or `skipped` |
| `error`                                | error message if something went wrong                                                                 |
| `conversion_time`                      | seconds needed for this report                                                                        |
| `pages`, `word_count`                  | size of the report                                                                                    |
| `detected_tables`, `detected_pictures` | number of tables and pictures found                                                                   |
| `mean_grade`, `low_grade`              | docling's quality grade for the conversion                                                            |
| `parse_score`                          | quality of the PDF's text layer; empty for scanned reports                                            |
| `ocr_score`                            | OCR quality; empty if no OCR was needed                                                               |

## How parsing works

- Reports are converted one at a time. A broken PDF is marked `failed` and the run continues.
- **OCR is applied only where needed:** pages without a text layer (scanned pages) and pages with a broken text layer are read with OCR. Normal digital pages keep their original text.
- A GPU is used automatically if one is available (needs a CUDA build of PyTorch, see [docling documentation](https://docling-project.github.io/docling/usage/gpu/)). Otherwise everything runs on the CPU.

The code is in `ghg_extraction/parsing.py`.

### Indexing

TBD.

### Retrieval

TBD.

### Generation

TBD.
