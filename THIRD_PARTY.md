# Sources and licenses

Application modules in `lecture2doc/` are newly written for this workspace. No upstream project runtime source is vendored into the package.

Research source snapshots are fetched from these MIT-licensed projects, at the revisions documented in `docs/RESEARCH.md` and `research/benchmark.json`. Their LICENSE files are stored alongside local snapshots and fetched by `research/benchmark_upstream.py --fetch`:

- hwanyong/analysis-video — Copyright (c) 2026 UHD — https://github.com/hwanyong/analysis-video
- eric-lf02/lecture-capture — see upstream LICENSE — https://github.com/eric-lf02/lecture-capture
- drpwchen/lecture-to-notes — see upstream LICENSE — https://github.com/drpwchen/lecture-to-notes
- kdavila/lecturemath — Copyright (c) 2021 Kenny Davila and Fei Xu — https://github.com/kdavila/lecturemath

Dependency distributions retain their own licenses and notices. System FFmpeg is executed, not bundled. Its build configuration and optional libx264 encoder need separate consideration if redistributing binaries. No AGPL research project has been added as an application dependency.

V1 optional OCR uses [RapidOCR 3.10.0](https://github.com/RapidAI/RapidOCR) (Apache-2.0), ONNX Runtime (MIT), and converted PaddleOCR PP-OCRv6 / Korean PP-OCRv5 models. Model attribution is described in [RapidOCR model licenses](https://github.com/RapidAI/RapidOCR/blob/main/MODEL_LICENSES.md) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE). Model files are downloaded separately to `models/ocr/`; their source URLs and SHA-256 hashes are preserved in the language manifests and C results. No model runtime source was copied into application modules.

Real example media and derived real-lecture comparisons:

David Jerison. **18.01 Single Variable Calculus. Fall 2006. Lecture 1: Rate of Change.** Massachusetts Institute of Technology: MIT OpenCourseWare. https://www.ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/resources/lecture-1-derivatives/

License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The local example is excerpted at 180–225 seconds, re-encoded, sampled/cropped for comparisons, and annotated with analysis results. These example assets retain that license and are not covered by the application code license. Real source video/model weights are local experimental inputs and excluded from source control.

AccessMath experiment data: the official AccessMath2015 / ICDAR 2017 video and annotation distribution, linked by [RIT DPRL](https://cs.rit.edu/~dprl/software.html). Original URLs and SHA-256 hashes are recorded in `experiments/accessmath/dataset-provenance.json`. The downloaded annotation archive includes `gpl-3.0.txt`, preserved under `datasets/accessmath/annotations/`. Dataset assets and derived comparison images remain separate from application code. No AccessMath runtime source code was incorporated into the application. The local research use, scope, and evaluation limitations are documented in `docs/ACCESSMATH.md`.
