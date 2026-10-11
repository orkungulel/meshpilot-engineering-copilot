# MeshPilot: running it

Placeholders in the commands:
- `<DATASET>`: your copy of the Fusion 360 Gallery segmentation data, converted by the thesis pipeline.
- `<RUN_RECORDS>`: the training run's record folder (`train_args.json`, `history.json`).
- `runs_fusion_dn_xyz` and its `checkpoint_last.pt`: a trained checkpoint. None of these are in the repository.

MeshPilot is a second track. It shows the per-face classes predicted by a
DiffusionNet model from my MSc thesis, in a 3D viewer next to the model's
provenance and limitations.

- **So far:**
  - real inference on held-out Fusion 360 Gallery meshes;
  - a check that this inference matches the training run;
  - the viewer;
  - a chat with one geometry tool.
- **Classes:** they are Fusion 360 modelling operations (extrude, cut,
  fillet, chamfer and revolve faces), not machining features.

It needs these files, all kept out of git:

- the thesis checkpoint, `runs_fusion_dn_xyz/checkpoint_last.pt`;
- the copy of the `diffusion_net` library used in training, in
  `meshpilot_assets/src/diffusion_net`;
- the converted dataset, with the run's recorded `split.json`.

The Fusion 360 Gallery licence allows non-commercial research use only and
forbids redistribution, so no mesh is committed.

On Windows (PowerShell, tested with Python 3.13), from the repository folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,meshpilot]"

# 1. Predict one test mesh (about 10 s on a CPU)
.venv\Scripts\python -m meshpilot.predict --checkpoint runs_fusion_dn_xyz\checkpoint_last.pt --meshes-dir <DATASET>\converted\all_seg\meshes --stem 23910_316134bd_3 --split <DATASET>\converted\all_seg\split.json --class-names <DATASET>\segment_names.json

# 2. Check the inference path against the training run (about 2 minutes)
.venv\Scripts\python -m meshpilot.verify --checkpoint runs_fusion_dn_xyz\checkpoint_last.pt --cache <DATASET>\converted\all_seg\cache\test_canonical_keig64_zb1_nc8.pt --history <RUN_RECORDS>\runs_fusion_dn_xyz\history.json --split <DATASET>\converted\all_seg\split.json --meshes-dir <DATASET>\converted\all_seg\meshes --stem 23910_316134bd_3 --class-names <DATASET>\segment_names.json

# 3. Open the viewer at http://127.0.0.1:8001
.venv\Scripts\python -m uvicorn meshpilot.api:app --port 8001
```

- **`predict`** writes four files to `meshpilot_assets/outputs/`:
  - `prediction.json`: the labels with their hashes, library versions and
    preprocessing;
  - `prediction.ply`;
  - `ground_truth.ply`;
  - `report.md`.

  How it makes a prediction:
  - each step repeats the thesis training script;
  - the mesh reader's face order is checked against the OBJ file;
  - the checkpoint is loaded with `weights_only=True`;
  - the mesh must be in the test list of the run's recorded split;
  - the scores in `report.md` are for that one mesh, not the model's
    benchmark performance.
- **`verify`** reads the operators and labels that training cached for every
  test mesh. It checks two things:
  - that, for one mesh, the operators and predictions computed here match
    the cached ones;
  - that, over the whole cached test split, the checkpoint reproduces the
    mIoU, accuracy and per-class IoU recorded in the run's `history.json`.

  It writes `verify.json` and exits with 1 if anything disagrees.
- **The viewer** shows these predictions; it runs no model itself. It has:
  - layers for the original mesh, the predicted classes and the dataset's
    labels;
  - a class legend that highlights the faces predicted as a class. The rest
    becomes translucent, or hidden with "Isolate";
  - the model's provenance;
  - the test-split IoU from `verify.json`, labelled as context, not
    confidence;
  - single-mesh scores and the limitations.

  It is separate from the document RAG's API, so each runs without the
  other's files.
- **The chat**, in the viewer, answers questions about the mesh with one
  geometry tool: "show the faces predicted as <class>". Three preset
  questions name classes the model has.
  - **Highlight:** the tool's structured result drives it, so the faces shown
    come from code, not from the model's text.
  - **Patches:** faces that share an edge and have the same predicted class.
    They are not validated feature counts.
  - **Evidence:** document citations and model predictions are shown in
    separate boxes.
  - **Needs:** a provider key in `.env`, which is gitignored. The viewer
    works without one.
    - For Claude: `ANTHROPIC_API_KEY`.
    - For Perplexity's Agent API: `PERPLEXITY_API_KEY` and
      `MESHPILOT_CHAT_PROVIDER=perplexity`. The model is set with
      `PERPLEXITY_MODEL` and defaults to `openai/gpt-6-luna`.
    - Only our function tools are sent to Perplexity: never its web search
      or other built-in tools (DECISIONS 25).
    - The chat panel shows which provider is active.
  - **Documents:** searched only when the RAG index is built.
  - **Cost:** each question makes at most `MESHPILOT_CHAT_MAX_STEPS` API
    requests (default 3), and the last one is made without tools.
  - **Testing:** the tests use a scripted client, so they need no key.
  - **Smoke test:** `python -m meshpilot.smoke` shows the three scenarios and
    the exact text it would send.
    - Add `--run` to send them.
    - Each request is checked against a request count and a worst-case
      dollar limit before it is sent (defaults: 6 and $0.25).
    - Token usage and cost are recorded in `meshpilot_assets/outputs/smoke/`.

- **The chat's documents:** the chat cites one project document, the
  MeshPilot notes in `docs/meshpilot/demo-corpus/`. They cover the class
  meanings, the model's provenance and scores, its limitations, an example
  review and the evidence rules. The figures in them are filled from the
  verify and predict records at build time.

  ```powershell
  .venv\Scripts\python -m meshpilot.corpus build --prediction runs_fusion_dn_xyz/checkpoint_last/23910_316134bd_3
  .venv\Scripts\python -m meshpilot.corpus search "What does the CutEnd class mean?"
  ```

  The build:
  - checks the records against the checkpoint, split, mesh and labels;
  - writes a 9-page PDF, its manifest and the `MP1` index to
    `data/meshpilot/` (gitignored);
  - checks the PDF's text, tables, links and page numbers;
  - uses `BAAI/bge-small-en-v1.5` pinned to one revision (about 134.5 MB,
    downloaded on the first build).

  The geotechnical corpus and its R1–R3 indexes are not touched.

Demo walkthrough: [docs/meshpilot/demo.md](demo.md).
Decisions: [DECISIONS.md](../../DECISIONS.md) entries 22 to 24.
