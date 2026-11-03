# C OCR / MER survey assets

Checked: 2026-11-02. Main report: [C_SURVEY.md](../../docs/C_SURVEY.md).

Research notes, pinned source files, model cards, and preprocessing examples for C.

- `source-index.json`: nine repositories, pinned commit SHA, selected files, upstream URL and SHA-256. Each source snapshot retains its upstream LICENSE. Some repositories were inspected through README/configuration only; the main report distinguishes them.
- `model-card-index.json`: eleven pinned model cards, metadata licenses and file hashes. Additional Pix2Text cards are linked directly in the report; they are not included in this eleven-card index.
- `sources/`, `model-cards/`: small public-source snapshots. 
- `recognition-accuracy.json`, `accuracy.html`: published whole-expression accuracy, source tables, and proposed adoption criteria.
- `example-index.json`, `examples/index.html`: six manually selected native-resolution AccessMath source crops, RGB/grayscale/adaptive-threshold variants, source coordinates/time/hash. 

## Reproduce

From the repository root, use a new copy/output folder if retaining the historical snapshot is required. Collection scripts contact official GitHub/Hugging Face endpoints and resolve the current HEAD; rerunning later may collect different revisions. The JSON indices are the authoritative record of the revisions used in this survey.

```bash
.venv/bin/python research/ocr-survey/collect_sources.py
.venv/bin/python research/ocr-survey/collect_details.py
.venv/bin/python research/ocr-survey/collect_integrations.py
.venv/bin/python research/ocr-survey/build_examples.py
.venv/bin/python research/ocr-survey/build_accuracy.py
```

`collect_sources.py` writes the initial six-repository index. Run the detail and integration steps afterward to obtain the full nine-repository index. To retrieve precisely the preserved sources, use each indexed URL (which includes a commit/revision) and check its SHA-256 rather than resolving HEAD again.

Next: compare recognition quality and runtime on the same lecture crops.
