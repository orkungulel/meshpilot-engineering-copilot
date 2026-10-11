# Engineering document QA and MeshPilot

A research prototype in two parts:

- **Document QA (RAG):** answers questions from public engineering manuals
  and cites document and page. It comes with a retrieval-evaluation harness
  and a small tool-using agent.
- **MeshPilot:** shows, on a CAD mesh, the class that a DiffusionNet model
  from my MSc thesis predicts for each triangle. A chat answers questions
  about the mesh, and keeps model predictions apart from document
  citations.

**Status: research prototype.**
- It is not validated for engineering decisions and is not a compliance
  check.
- Model weights, datasets and documents are not in this repository (see
  [Licences and third-party assets](#licences-and-third-party-assets)).

## The problem

Engineers look up answers in long manuals. More and more, they also look at
machine-learning predictions on CAD geometry. Both need provenance: which
page says so, which model predicted it, and how good that model is.

This project keeps that provenance visible:
- every answer cites its document and page;
- every prediction carries its checkpoint and its measured test scores;
- document evidence and model evidence are never presented as each other.

## Architecture

```
Browser: MeshPilot viewer (one static page, plain WebGL)
  mesh view · prediction / ground-truth / original layers · class highlight and isolate
  provenance and verification panel · chat, with the active provider shown (Claude, Perplexity or MOCK)
        │ HTTP on localhost
meshpilot.api (FastAPI): serves saved predictions after checking their hashes
  /chat → meshpilot.agent (the document agent's loop, with tools):
           ├─ show_predicted_class → meshpilot.regions: local, deterministic face ids → the highlight
           ├─ search_documents     → index MP1: local FAISS over the MeshPilot notes, cited by page
           └─ provider: Anthropic SDK | Perplexity Agent API adapter | scripted MOCK
Offline commands:
  meshpilot.predict  DiffusionNet inference with the thesis checkpoint (restricted loading, matched preprocessing)
  meshpilot.verify   reruns the checkpoint on the training run's cached test split and checks its recorded scores
  meshpilot.corpus   builds the notes PDF and its index from the verify and predict records
  meshpilot.smoke    bounded live chat tests with request and cost limits
rag/: document QA. PDF → page-bounded chunks → embeddings → FAISS → cited answer or agent; eval/ harness; FastAPI; Docker.
```

## What works now

| Capability | Evidence |
|---|---|
| Real DiffusionNet inference with the thesis checkpoint: restricted loading, preprocessing that matches training, deterministic output | `meshpilot.verify` reproduces the training run's recorded test scores ([record](eval/results/meshpilot/2026-10-11_verify_fusion_xyz.json)) |
| 3D viewer: three layers, highlighting and isolating a class, provenance and verification panel | `meshpilot/static/viewer.html`, `tests/test_meshpilot_api.py` |
| Chat with a geometry tool whose structured output drives the highlight, with document citations kept apart | `meshpilot/agent.py`, tests, and the live smoke runs below |
| Perplexity provider: function tools only, no web search, bounded requests | `meshpilot/perplexity.py`, `meshpilot/smoke.py`, and the live smoke runs below |
| MeshPilot notes: a project document whose figures are filled from records, built into a page-checked PDF and a pinned local index | `meshpilot/corpus.py`, [DECISIONS 26](DECISIONS.md) |
| Document QA: page citations, a retrieval-evaluation harness, an agent, an API and a Docker image (built in CI) | `rag/`, `eval/run_eval.py`, [docs/rag.md](docs/rag.md) |
| Repository rules: no key, PDF, checkpoint or `.env` file can be committed | `tests/test_repo_hygiene.py` |

The Claude provider and the document-QA answers have run only against
scripted clients in tests. No real Claude call has been made yet.

## Measured results

All from committed records in `eval/results/`.

- **MeshPilot model** (thesis DiffusionNet, final checkpoint, 1,000 held-out
  Fusion 360 Gallery test meshes):
  - mean IoU 0.2375 and triangle accuracy 0.5507, identical to the training
    run's recorded values;
  - per-class IoU from 0.06 (Fillet) to 0.50 (ExtrudeSide), so it is a weak
    model;
  - [record](eval/results/meshpilot/2026-10-11_verify_fusion_xyz.json).
- **Live chat checks** (owner-approved and bounded; Perplexity Agent API with
  `openai/gpt-6-luna`), all three passed:
  - a supported class led to a real tool call and the correct highlight;
  - an unsupported class led to a clear refusal, with nothing highlighted;
  - a document question led to a search and a correct page citation.

  Together they made 5 requests, cost $0.00058 as reported by Perplexity, and
  are only a smoke test, not an evaluation
  ([record](eval/results/meshpilot/2026-10-11_smoke_perplexity.json)).
- **Document QA:** not yet measured. The evaluation set has not been
  written, so there are no RAG numbers.

## Quickstart

### Key-free MOCK demo (fresh clone: no model, no data, no API key)

On Windows, clone to a short path such as `C:\src\meshpilot-engineering-copilot`.
Otherwise `pip install` can fail on a long file name inside the `anthropic`
package, which runs past Windows' 260-character path limit.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m meshpilot.mock_demo
```

Open http://127.0.0.1:8001. Everything in this demo is labelled MOCK:
- the mesh is a synthetic stepped shaft generated in code;
- its labels come from a geometric rule, not a model;
- the chat is a scripted client that calls the real geometry tool.

### Tests

```powershell
.venv\Scripts\python -m pytest
```

They need no API key, model weights, PDFs or datasets.

### Optional local setup: real providers and your own assets

The MOCK demo and the tests need none of this.

- **Real chat providers:**
  - Put your own `ANTHROPIC_API_KEY` or `PERPLEXITY_API_KEY` in the
    gitignored `.env` (see `.env.example`).
  - The chat calls a provider only when its key is set, and the calls are
    billed to you.
  - Without a key, the chat is off and the rest of the viewer works.
- **Document QA:** see [docs/rag.md](docs/rag.md). It covers the corpus,
  index, API, Docker and evaluation.
- **MeshPilot with a trained checkpoint:** see
  [docs/meshpilot/usage.md](docs/meshpilot/usage.md). It needs three things,
  none of them included:
  - a checkpoint;
  - the `diffusion_net` source used in training;
  - the converted dataset.

## Limitations

- **A research prototype.** Its predictions are not validated for
  engineering decisions and are not compliance decisions.
- **The model is weak and uneven by class**, and whole regions can be wrong.
  Test-split IoU is not a confidence for any single part.
- **Narrow training conditions:**
  - one checkpoint, trained on a Fusion 360 Gallery subset, with no
    augmentation;
  - other poses, tessellations and data sources are untested.
- **Per-triangle predictions.** They are not B-rep faces, and a patch of
  same-class triangles is not a feature instance.
- **Classes are CAD modelling operations**, not machining features or
  manufacturing plans.
- **Retrieval scores do not show whether a question is answerable**; the
  model must judge from the cited excerpts.
- **Few live calls so far.** Five live chat requests were made, and the
  Claude path and the RAG evaluation have not run against a real API.
- **Local use only.** It is a demo, not a production deployment.

## Future ideas (not built)

These are not part of v1:
- a review or ticket workflow;
- a CAD plugin;
- surrogate models;
- reranking;
- new embedding models or a retrieval redesign;
- more datasets or checkpoints.

## Licences and third-party assets

**This repository contains original work only:** source code, tests,
documentation and measurement records.
- **Licence:** the [Apache License 2.0](LICENSE), for that original work
  only. See also [NOTICE](NOTICE).
- **Not covered:** the Apache-2.0 licence does not cover datasets,
  checkpoints, generated outputs or third-party assets. None of them is
  included, and each remains subject to its own licence or terms.

**PyMuPDF: AGPL-3.0 or commercial.**
- **Where it is used:** the PDF/RAG pipeline uses PyMuPDF, which is
  distributed under AGPL-3.0 or a commercial licence from Artifex.
- **How it is installed:** it is a required dependency, installed from PyPI
  and not part of this repository.
- **Where it is loaded:** in the document QA code, and in the MeshPilot
  app, which loads it at start-up through its notes module, even in the
  MOCK demo.
- **Before redistribution or deployment, review the applicable licence.**
  That includes publishing a Docker image built from this repository, or
  running it as a network service.
- **No Docker image is published,** because the image would include
  PyMuPDF. The `Dockerfile` is for local builds only.

### Dependencies (installed from PyPI, not included)

| Package | Licence (from its package metadata) | Needed for |
|---|---|---|
| pymupdf | AGPL-3.0 or Artifex commercial licence | PDF text extraction and the MeshPilot notes PDF |
| anthropic | MIT | the optional Claude provider |
| fastapi, uvicorn | MIT, BSD-3-Clause | the APIs and the viewer |
| faiss-cpu | MIT | the vector index |
| numpy | BSD-3-Clause, plus permissive licences of bundled parts | arrays |
| python-dotenv | BSD-3-Clause | reading `.env` |
| sentence-transformers, transformers (optional `embed`) | Apache-2.0 | embeddings |
| torch (optional `embed`, `meshpilot`) | BSD-3-Clause, plus permissive licences of bundled parts | embeddings and inference |
| potpourri3d, robust_laplacian, scikit-learn, scipy (optional `meshpilot`) | MIT, MIT, BSD-3-Clause, BSD-3-Clause | DiffusionNet preprocessing |
| httpx, pytest (optional `dev`) | BSD-3-Clause, MIT | the tests |

Two indirect dependencies, certifi and tqdm, are MPL-2.0 (tqdm also MIT).
All the others are permissive. Check again for the versions you install.

### External assets (not included, each under its own terms)

| Asset | Role here | Terms |
|---|---|---|
| Fusion 360 Gallery segmentation dataset (Autodesk) | training and test data of the thesis model | non-commercial research use only; no redistribution |
| Thesis DiffusionNet checkpoints | MeshPilot inference | not published, and not licensed |
| [diffusion_net](https://github.com/nmwsharp/diffusion-net) | model code, imported at run time from your own copy | MIT |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | embeddings, downloaded when an index is built | MIT |
| FHWA and USACE manuals | the document QA corpus; `corpus/manifest.csv` holds only their titles and URLs | US federal publications; check each one for third-party material |
| Claude API, Perplexity API | optional chat providers, used only with your own key | each provider's terms; calls are billed to you |

- **What you generate:** predictions, indexes, PDFs and recordings made from
  these assets may remain subject to the assets' terms.
- **The key-free MOCK demo** uses a synthetic mesh generated in code, and
  needs none of these assets.

## Documentation

- [DECISIONS.md](DECISIONS.md): every design decision, with the
  alternatives rejected.
- [docs/rag.md](docs/rag.md): the document QA guide.
- [docs/meshpilot/usage.md](docs/meshpilot/usage.md): running MeshPilot.
- [docs/meshpilot/demo.md](docs/meshpilot/demo.md): the demo walkthrough.
