# GHG Extraction

Extracts greenhouse gas (GHG) emission figures from company reports (PDF).

The first step, **parsing**, is done: every report PDF is converted into Markdown and JSON with [docling](https://github.com/docling-project/docling). Later steps (indexing, retrieval, generation) are planned and prepared in `main.py`.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

This creates the `.venv` folder and installs all packages from `uv.lock`.

## Data

Put the reports into `data/reports/`, with one folder per company. A company folder can hold reports of several years:

```
data/
├── reports/
│   ├── Allianz/
│   │   └── Allianz_2022.pdf
│   └── hugo boss/
│       ├── hugo boss_2019.pdf
│       └── hugo boss_2020.pdf
└── gold_data/          # reference values for evaluation (used later)
```

- File names should follow `<company folder name>_<year>.pdf`. Other names are still parsed, but a warning is logged.
- `.pdf` and `.PDF` are both accepted.
- `data/reports/` is not stored in git.

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

The `output/` folder is created automatically in the project root (not stored in git):

```
output/
├── converted/
│   └── <company>/
│       ├── <company>_<year>.md     # text of the report as Markdown
│       └── <company>_<year>.json   # full docling document (structure, tables, positions)
├── logs/
│   └── pipeline_<timestamp>.log    # one log file per run
└── summary_<timestamp>.csv         # one row per report, one file per run
```

The summary CSV has one row per report with, among others:

| Column | Meaning |
|---|---|
| `status` | `success`, `partial_success`, `warning` (fewer than 100 words, check manually), `failed` or `skipped` |
| `error` | error message if something went wrong |
| `conversion_time` | seconds needed for this report |
| `pages`, `word_count` | size of the report |
| `detected_tables`, `detected_pictures` | number of tables and pictures found |
| `mean_grade`, `low_grade` | docling's quality grade for the conversion |
| `parse_score` | quality of the PDF's text layer; empty for scanned reports |
| `ocr_score` | OCR quality; empty if no OCR was needed |

## How parsing works

- Reports are converted **one at a time**. A broken PDF is marked `failed` and the run continues.
- **OCR is applied only where needed:** pages without a text layer (scanned pages) and pages with a broken text layer are read with OCR. Normal digital pages keep their original text.
- A GPU is used automatically if one is available (needs a CUDA build of PyTorch). Otherwise everything runs on the CPU.

The code is in `ghg_extraction/parsing.py`.
