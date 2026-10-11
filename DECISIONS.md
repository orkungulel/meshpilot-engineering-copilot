# Design decisions

Each entry records what was chosen, the main alternatives that were rejected
and why, and what would change the decision. Entries are added when a decision
is made, so they are in chronological order.

## 1. Plain Python, no RAG framework

- **Chosen:** small modules written directly against PyMuPDF,
  sentence-transformers, FAISS and the Anthropic SDK.
- **Rejected:** LangChain or LlamaIndex. They would save some lines, but they
  hide chunking, prompting and the agent loop behind abstractions, and those
  are exactly the parts I need to explain and change. They also bring a large
  dependency tree into a small prototype.
- **Revisit if:** the system grows to several data sources or retrieval
  strategies, where a framework's integrations would save real work.

## 2. Exact dependency pins

- **Chosen:** every dependency pinned to an exact version in `pyproject.toml`.
- **Why:** evaluation numbers depend on library behaviour, PDF text extraction
  above all. With version ranges, a fresh install could silently change the
  corpus text and therefore the results.
- **Rejected:** version ranges. Easier upgrades, but results would not be
  reproducible.

## 3. Configuration and the API key

- **Chosen:** settings come from environment variables with a `RAG_` prefix:
  `RAG_MODEL` (default `claude-sonnet-5-5`) writes answers, `RAG_JUDGE_MODEL`
  (default `claude-opus-5-5`) judges them in the evaluation. The Anthropic SDK
  reads the key from `ANTHROPIC_API_KEY`. Locally it lives in a gitignored
  `.env`; Docker receives it at run time (`--env-file`) and it is never baked
  into the image; CI needs no key.
- **Why the prefix:** Claude Code itself reads `ANTHROPIC_MODEL`. A
  project-specific name avoids picking up a value meant for another tool.
- **Rejected:** hard-coding the model (the evaluation compares configurations,
  so they must be switchable without code changes); a secrets manager (too
  much for a prototype with one secret).

## 4. Tests need no model weights, documents or API key

- **Chosen:** unit tests use PDFs generated on the fly, a fake embedder and a
  fake Claude client. CI installs the package without the embedding model.
- **Why:** tests stay fast and deterministic, and CI needs no secrets and no
  large downloads. The real embedding model and Claude are exercised by index
  builds and by `eval/run_eval.py`, whose results are committed.
- **Rejected:** downloading the real model in CI. Slower, can fail on network
  or rate limits, and still would not test answer quality, which is the
  evaluation's job.

## 5. Corpus files never enter the repository

- **Chosen:** PDFs live in `data/`, which is gitignored. A test fails the
  build if any `.pdf` or `.env` file is ever tracked.
- **Why:** licensing hygiene, repository size, and the rule that no personal or
  employer documents are committed. The corpus is rebuilt from a committed
  manifest instead.

## 6. Page numbers and PDF text extraction

- **Chosen:** a page is identified by its 1-based index in the PDF (the number
  a PDF viewer shows), not by its printed page number. The printed label is
  kept for display only. Text comes from PyMuPDF's plain-text mode with
  ligatures expanded, so "ﬁeld" in the PDF matches "field" in a question.
- **Why:** printed numbers restart after the front matter and repeat across
  volumes. The PDF index is unambiguous, so a question's expected page and the
  system's citation are compared on the same scale.
- **Not done (yet):** rejoining words hyphenated across line breaks (a regex
  would also merge real compounds such as "cross-section"), removing repeated
  page headers and footers, and OCR for scanned pages (documents without a
  text layer are left out of the corpus instead). Revisit if retrieval misses
  turn out to come from these.

## 7. Chunking: overlapping word windows inside one page

- **Chosen:** windows of N words, a new one every 0.8 N words (20% overlap),
  never crossing a page. Two sizes are fixed for the evaluation before any
  result exists: 300 words with 60 overlap (the default) and 150 words with 30.
- **Why these sizes:** the default embedding model (bge-small-en-v1.5) reads at
  most 512 tokens and silently ignores the rest. By the usual rule of thumb of
  about 1.3 tokens per English word, 300 words is roughly 400 tokens, which
  leaves headroom for numbers and units (they split into many tokens). The
  index build measures how many chunks actually exceed the limit. The 150-word
  variant tests whether smaller, more focused chunks find the right page more
  often. With 20% overlap, a fact cut by one window's boundary still appears
  whole in the neighbouring window.
- **Rejected:**
  - Chunks that span pages: each chunk would need a page range, which makes
    citations and page-level scoring ambiguous.
  - One chunk per page: a full page of dense technical text can exceed the
    512-token limit and be truncated.
  - Sentence-aware or "semantic" chunking: more code, and no evidence yet that
    it helps. Worth trying if the evaluation shows misses at chunk boundaries.
  - Sizes counted in tokens: exact, but ties the chunker to one tokenizer.
    Words plus a measured truncation rate is simpler.
- **Known cost:** a sentence that runs across a page break is split in two.

## 8. Corpus: US federal geotechnical guidance, rebuilt from a manifest

- **Chosen:** FHWA Geotechnical Engineering Circulars and US Army Corps of
  Engineers engineering manuals on foundations, retaining structures, slopes
  and site investigation. They are US federal government publications and
  guidance manuals, not standards (GEC 13, for example, states that it does
  not constitute a standard, specification or regulation).
- **Why:** they are what a practising engineer actually looks things up in.
  They are full of numbers and tables, which gives the calculator tool real
  work. And a single theme makes retrieval realistically hard, because many
  documents share the same vocabulary.
- **Rejected:**
  - Structural failure investigations (NTSB, NIST): famous cases are partly
    in the model's training data, so a correct answer would not prove that
    retrieval worked.
  - Open-access research papers: clear licences, but not what engineers
    consult in practice, and two-column layouts and reference lists add
    extraction noise.
- **Reproducibility:** `corpus/manifest.csv` is committed with each file's
  SHA-256 and page count, pinned at the first download. A changed file fails
  loudly, because a new edition can move the pages that evaluation questions
  point to. A download must also start with a `%PDF-` header: without that
  check, PyMuPDF quietly "repairs" an HTML error page into a one-page PDF.
- **Downloading:** the FHWA, ROSA P and DTIC servers answered HTTP 403 to
  automated downloads from GitHub's runners, and the mirror for EM 1110-2-1902
  had an expired TLS certificate. Rather than disguise the downloader as a
  browser or switch off certificate checks, the PDFs are saved by hand
  (`rag.cli sources` lists file names and URLs) and `download --pin` verifies
  and pins them.

## 9. Embedding model: bge-small-en-v1.5, with all-MiniLM-L6-v2 as comparison

- **Chosen:** `BAAI/bge-small-en-v1.5` (33M parameters, 384-dimensional
  vectors, 512-token input limit, MIT licence), run locally on CPU through
  sentence-transformers. Queries get the instruction prefix the bge models were
  trained with; passages do not.
- **Why:** small enough to embed the whole corpus on a laptop CPU, a token
  limit that fits 300-word chunks, and it ranks higher than MiniLM on the
  public MTEB retrieval benchmark.
- **Comparison (R3):** `sentence-transformers/all-MiniLM-L6-v2` (22M
  parameters, 384 dimensions, 256-token limit), the most common baseline.
  Because of its 256-token limit it is only paired with 150-word chunks.
- **Rejected:** larger models such as bge-base or bge-large (stronger on
  benchmarks but several times slower on CPU; worth trying only if retrieval
  turns out to be the bottleneck); a paid embedding API (excluded by the brief,
  and it would send the corpus to a third party); multilingual models (the
  corpus is English).
- **Reproducibility:** each index's `build_info.json` records the model, the
  query prefix, library versions and how many chunks exceeded the token limit.

## 10. Vector store: FAISS, exact search

- **Chosen:** FAISS `IndexFlatIP` over normalised vectors, which is exact
  cosine-similarity search, saved as three files: the FAISS index, the chunks
  as JSON lines, and the build record.
- **Why:** brute-force search over tens of thousands of 384-dimensional vectors
  takes milliseconds. Exact search also means every retrieval miss comes from
  the embeddings or the chunking, not from an approximate index.
- **Rejected:**
  - Chroma: a database with its own persistence and metadata store. Nothing
    here needs one: the index never changes after the build and nothing is
    filtered by metadata.
  - Approximate indexes (HNSW, IVF): only pay off at millions of vectors, and
    their recall loss would blur the evaluation.
  - Plain numpy: would work at this size. FAISS costs one dependency and makes
    a later switch to an approximate index a small change.
- **Safety check:** an index refuses to load with a different embedding model
  from the one it was built with, because the query vectors would not be
  comparable with the stored ones and the ranking would be silently wrong.

## 11. No reranker, until the numbers ask for one

- **Chosen:** the top 5 chunks by cosine similarity go straight to the answer
  model. There is no cross-encoder reranking step.
- **Why:** a reranker adds a second model, latency and complexity, and whether
  it could help is measurable before building it. It can only help when the
  right page is among the candidates it would reorder. So the retrieval
  evaluation reports recall@20 next to recall@5. If recall@20 is much higher
  than recall@5, the right page is found but ranked too low, and a reranker over
  the top 20 is the next step. If recall@20 is also low, a reranker cannot help;
  the chunking or the embedding model is the problem.
- **Rejected for now:** a cross-encoder reranker over the top 20 to 50
  candidates.

## 12. Evaluation questions: format and checks

- **Chosen:** one JSON line per question with the expected document and
  1-based PDF page, a reference answer, a type, and who wrote it (`human`, or
  `claude-drafted-human-verified`). At most 3 questions per document, so no
  single manual dominates. An `also_acceptable` page list exists only for a
  fact the corpus genuinely states twice.
- **Checks:** CI validates the structure against the committed manifest.
  Locally, with the PDFs present, the validator warns when a number in the
  reference answer is not on the stated page (this catches a printed page
  number used instead of the PDF page) and when a question copies 8 or more
  consecutive words from its page (copied wording makes retrieval look better
  than it is).
- **Rejected:** questions generated automatically from chunks by an LLM. They
  tend to reuse the chunk's wording, which inflates retrieval scores, and the
  expected answers would come from a model rather than an engineer. Also
  rejected: several expected pages by default, which makes scoring lenient and
  ambiguous.

## 13. Retrieval metrics and how results are reported

- **Chosen:** recall@5 as the headline: the share of questions where a chunk
  from the expected page is among the top 5. With one expected page per
  question it equals the hit rate at 5. Diagnostics: recall@1, recall@20 (the
  reranker ceiling, entry 11) and document-level recall@5.
- **Uncertainty:** each rate is reported with a 95% Wilson interval. Two
  configurations are compared with an exact McNemar test, because both run on
  the same questions and only the questions where they disagree carry
  information.
- **Rejected:** MRR or nDCG as the headline (harder to explain and they add
  little with one relevant page per question); the normal-approximation
  interval (misbehaves at n = 30 and at rates near 0 or 1); comparing two
  percentages as if they came from independent samples (ignores the pairing,
  so it is less able to detect a real difference).
- **Provenance:** every run folder records the git commit and whether there
  were uncommitted changes, the question file's SHA-256 and the index's build
  record (configuration, model, document hashes). Only runs from a clean
  commit go into the README. Earlier result folders that are not committed yet
  do not count as uncommitted changes.

## 14. Answers: numbered sources, structured output, no citation, no answer

- **Chosen:** the top 5 chunks go to the model as numbered sources (S1 to S5)
  with document title and page, followed by the question. The model replies
  with JSON matching a schema: `answerable`, `answer`, and `citations`
  restricted to the source ids it was shown. The code maps ids back to
  document and page, drops (and records) any id that was not shown, and does
  not return an answer without a valid citation: the reply becomes "The
  retrieved documents do not contain an answer I can cite."
- **Why:** structured output makes citations data rather than text to parse.
  The schema plus the check in code means the system can only cite pages it
  actually retrieved, and refusing uncited answers turns "every answer cites a
  document and page" from a hope into a guarantee.
- **Abstention:** the system prompt forbids outside knowledge and asks the
  model to set `answerable` to false when the sources lack the answer. The
  unanswerable question set measures how well that works.
- **Rejected:**
  - Free-text answers with `[1]`-style markers parsed by a regex: brittle, and
    nothing stops the model from citing a source that does not exist.
  - The API's built-in document citations: they also guarantee valid sources
    and add quoted spans, but they cannot be combined with structured output,
    so the answerable flag would have to be parsed from text.
  - Abstaining below a retrieval-score threshold: the threshold would have to
    be tuned on labelled questions, and cosine scores are not comparable
    across queries.
- **Model settings:** `RAG_MODEL`, default `claude-sonnet-5-5`, with thinking
  and effort left at the model's defaults (not tuned). Server-side fallback is
  on: if a safety classifier declines a request, the API re-runs it on the
  fallback model Anthropic recommends, and every answer records which model
  actually served it. Sonnet 5.5 does not accept a temperature setting, so
  answers can vary between runs; the evaluation therefore repeats each
  question.

## 15. Judging answers: a different model, a pass/fail rubric, checked before use

- **Chosen:** `claude-opus-5-5` (`RAG_JUDGE_MODEL`) grades each answer against
  the engineer's reference answer. Correct only if it gives the key facts with
  matching numbers, contradicts nothing, and actually answers; saying the
  documents lack the answer counts as incorrect on an answerable question. It
  replies with structured output, the reason before the verdict. It does not
  see the retrieved sources: it grades correctness, not faithfulness.
- **Why:** a different model from the one being graded avoids a model
  favouring its own style. A pass/fail decision with concrete conditions is
  more repeatable than a 1-to-5 scale. Reference answers come from an
  engineer, so the judge compares rather than relying on what it knows.
- **Checks before its numbers are used:** reference answers graded as if they
  were system answers must pass; an empty answer, "I don't know" and another
  question's reference answer must fail; and one full run is also labelled by
  hand without seeing the judge's verdicts (entry 21), with the agreement
  reported in the README.
- **Rejected:** exact or keyword matching (fails on paraphrases and unit
  conversions); grading every run by hand (the most trustworthy, but too slow
  to repeat for every configuration, so it calibrates the judge instead);
  using the answering model as its own judge.

## 16. Answer evaluation: repeated runs, unanswerable questions, a no-retrieval baseline

- **Chosen:**
  - Each question is answered 3 times, because Sonnet 5.5's answers cannot
    be made deterministic (it takes no temperature setting). A question counts
    as correct when it is correct in more than half of its runs, and the count
    for every single run is reported too, so run-to-run variation stays visible.
  - About 10 in-domain questions the corpus does not answer
    (`eval/unanswerable.jsonl`) test the "say so instead of guessing"
    requirement. Without them, a system that never abstains would look perfect.
  - Baseline A0 asks the same questions with no retrieval. If it scores close
    to the retrieval runs, the questions test the model's memory, not
    retrieval.
  - Answers served by a model other than the one requested, refusals,
    truncated or unparseable replies and API errors go to `errors.jsonl` with a
    failure class and are not scored: a failed call is not a wrong answer.
- **Why a majority vote:** it keeps one yes/no per question, so the same
  Wilson intervals and McNemar comparison work for answers as for retrieval.
- **Rejected:**
  - Pooling all runs as if they were independent questions: 90 runs of 30
    questions are not 90 independent samples, and the interval would claim
    more precision than there is.
  - A single run: one unlucky sample could decide a comparison.
  - A bootstrap interval over questions: better suited to averaged scores, but
    one more method to explain for little gain at this size.
- **Cost control:** a pilot runs on the first 5 questions once (`--limit 5
  --reps 1`). Its recorded token usage gives the cost of a full run before one
  is started.

## 17. API: FastAPI, one shared service, fail fast at startup

- **Chosen:** FastAPI with `POST /ask` and `GET /health`. The index, embedding
  model and Anthropic client are loaded once at startup and shared by every
  request. `/ask` runs the agent by default (`mode: "rag"` gives one search and
  one answer). The response carries the answer, the answerable flag, citations
  (document, title, page), the agent's tool calls and the model that
  answered. An invalid question gets 422; a model that gives no
  usable answer, or a failed API call, gets 502. Without an index or an API
  key the service refuses to start, with a message saying what to do.
- **Why plain `def` handlers:** the Anthropic client and FAISS are
  synchronous, and FastAPI runs plain handlers in a thread pool, so requests
  still overlap. Retrieval runs under a lock because the embedding model's
  tokenizer is not safe to share between threads; it takes milliseconds,
  while the model call, which takes seconds, runs in parallel.
- **Rejected:** async handlers with the async Anthropic client (more moving
  parts, and no load test asks for them yet); Flask (no request validation or
  generated API docs); loading the index per request (seconds per request).

## 18. Docker: CPU-only, model baked in, index and key supplied at run time

- **Chosen:** `python:3.11-slim` with CPU-only torch from the PyTorch package
  index. The embedding model is downloaded during the build, and the container
  runs with `HF_HUB_OFFLINE=1`. The index and PDFs are mounted read-only at run
  time, the API key comes from `--env-file`, and the container runs as an
  unprivileged user. Dependencies are installed before the code is copied, so
  code changes rebuild in seconds.
  - **How the dependencies are installed:** their pins are read from
    `pyproject.toml` with `tomllib`. The project itself is installed in place
    with `--no-deps` once `rag/` and `meshpilot/` have been copied.
  - **What the image holds for MeshPilot:** its code and viewer page, but no
    checkpoint, mesh or prediction. `.dockerignore` keeps those out of the
    build context.
- **Why:** the default torch wheel bundles CUDA libraries that a 33M-parameter
  model on CPU never uses. A baked-in model makes startup independent of
  Hugging Face. Keeping the index outside the image means one image serves any
  corpus build, and no document text or secret ends up in an image layer.
- **Rejected:** baking the index into the image (ties the image to one corpus
  build); downloading the model at startup (needs network on every start and
  can fail); a GPU base image (unnecessary at this size); empty placeholder
  packages to install the project before its code is copied. Every new
  package needed another placeholder, and the build broke when `meshpilot`
  was added.
- **How it is verified:** Docker is not available in the environment this was
  developed in, so a separate CI workflow builds the image whenever the
  Dockerfile or dependencies change. It checks that the embedding model loads
  with networking switched off, and that the API refuses to start without a
  key. It also checks that `meshpilot` imports with its viewer page, and that
  no checkpoint, mesh, prediction or `.env` file is in the image.

## 19. Calculator: a parsed expression, never eval()

- **Chosen:** the agent's calculator parses the expression into a Python
  syntax tree and evaluates only numbers, `+ - * / % **`, unary signs and
  parentheses. `^` is read as a power, as engineers write it. Arithmetic is
  done in floats, expressions are limited to 200 characters, and division by
  zero, overflow, or a complex or infinite result comes back to the model as
  an error message, not as a crash.
- **Why:** the expression is written by a model, which can be wrong or be
  steered by text inside a retrieved document. `eval()` would run any Python
  it was given. Floats keep the cost bounded: `9**9**9` overflows immediately
  instead of building an integer with hundreds of millions of digits.
- **Rejected:** `eval()` with restricted globals (known to be escapable); an
  expression library such as sympy or numexpr (more than arithmetic, and more
  dependencies); letting the model do the arithmetic itself (language models
  make arithmetic slips; the tool makes each calculation exact and logged).

## 20. Agent loop: one tool per step, at most 5 steps, tools off on the last

- **Chosen:** a hand-written loop over the Messages API with two tools,
  `search_documents` and `calculator`. Each step is one model call making at
  most one tool call (parallel tool calls are switched off). There are at most
  5 steps, and on the fifth the tools are switched off (`tool_choice: none`),
  so the model has to answer from what it has found. The final answer uses the
  same structured JSON and "no citation, no answer" rule as plain retrieval.
  Sources are numbered across searches (an excerpt found again keeps its id),
  and every cited id is checked against the sources the model was actually
  shown. Every tool call is logged and returned with the answer: step, tool,
  input and a short summary of the output.
- **Why these limits:** the cap bounds cost and latency at 5 model calls per
  question and stops a model that keeps searching. One tool call per step keeps
  the log readable and the step count meaningful. Switching the tools off on
  the last step guarantees an answer, or an explicit "not in the documents",
  instead of an error.
- **Rejected:**
  - The SDK's tool runner: it hides the loop, and the step cap and the
    last-step behaviour are exactly what I need to show and explain. It is
    also in beta.
  - Agent frameworks such as LangChain agents: same reason, plus a large
    dependency tree.
  - Running until the model stops by itself: unbounded cost and latency.
  - Raising an error when the cap is reached: an answer from the evidence
    already gathered is more useful.

## 21. Blind labels: checking the judge against an engineer

- **Chosen:**
  - **The sheet:** `label-sheet` writes `labels.md` into an answer run's
    folder. It starts with the judge's rubric, the same text the judge gets
    (`RUBRIC` in `rag/judge.py`). Then, for each question, it shows the
    question, the reference answer and the system's answer, with an empty
    `label (y/n):` line. It contains no verdicts and no reasons from the
    judge.
  - **Which answers:** one per question by default, from its first run that
    gave an answer; `--all-reps` labels every run. Answers saying the
    documents do not contain the answer are left out.
  - **What `agreement` reports:** agreement as k/n with a Wilson interval,
    Cohen's kappa, the two-by-two table of judge against engineer, and each
    disagreement with the judge's reason.
  - **Keeping the result:** the first result is written to `agreement.json`
    and never overwritten.
- **Why:**
  - **Same inputs:** the engineer sees exactly what the judge saw and grades
    by the same rubric. A disagreement then means the judge applies the
    rubric differently, not that one of them had more information.
  - **Kappa:** raw agreement flatters a judge when most answers are correct,
    because one that passes everything agrees on every correct answer. Kappa
    corrects for that.
  - **The table:** it shows the direction of the errors. A judge that is too
    lenient inflates the headline score; one that is too strict deflates it.
  - **One answer per question:** the runs of one question are not
    independent, so 90 labels would tell little more than 30 for three times
    the work.
  - **No abstentions:** the rubric makes them incorrect whoever grades them,
    so including them would only inflate the agreement.
  - **The first result:** once the judge's reasons have been read, a changed
    label is no longer blind.
- **Rejected:**
  - Showing the judge's verdict and asking whether the engineer agrees: that
    anchors the person on the judge's answer. `review.md` does this, and it is
    meant for the pilot.
  - A CSV filled in with Excel: depending on the computer's regional settings,
    Excel saves CSV files with semicolons instead of commas and in a legacy
    encoding, so the file would not reliably read back. A Markdown file opens
    in any editor, reads well on GitHub and is easy to check line by line.
  - Agreement alone, or kappa alone: agreement hides chance agreement, and
    kappa is hard to read without the counts behind it.

## 22. MeshPilot inference: the thesis checkpoint and library as they were trained

- **Chosen:**
  - **Checkpoint:** the final-epoch checkpoint of the Fusion360 XYZ run,
    loaded with `torch.load(weights_only=True)`.
  - **Library:** `diffusion_net` is imported only from the copy taken from
    the training cluster (`meshpilot_assets/src`). A copy imported from
    anywhere else is refused.
  - **Preprocessing:** every step repeats `diffusion_net_train.py`:
    potpourri3d's mesh reader, `normalize_positions`, `get_operators` with
    the run's `k_eig`, and the input features named in the checkpoint's own
    config. Those features are checked against the shape of the model's
    first layer.
  - **Operators:** computed afresh, never read from the library's cache.
  - **Meshes:** only meshes in the test list of the run's recorded split are
    predicted.
  - **Output:** in the PLY files, each face gets its own three vertices.
- **Why:**
  - **Final epoch, not best:** the best-epoch checkpoint was picked by its
    test score, so that score is optimistic. The final epoch was not chosen
    by looking at the test set.
  - **XYZ, not HKS:** the HKS run's recorded test score is far lower.
  - **One library copy:** another version of `diffusion_net` could compute
    different operators, and so different predictions.
  - **`weights_only`:** it loads only tensors and plain containers. A full
    unpickle can run any code stored in the file.
  - **No operator cache:** the library's cache opens `.npz` files with
    `allow_pickle=True`, so it is skipped for the same reason.
  - **Face order:** the labels file, the PLY and the viewer all count faces
    in the OBJ's order. The reader's face order is therefore checked against
    the file.
- **Rejected:**
  - **The best-epoch checkpoint:** it was chosen on the test set.
  - **Upstream diffusion-net from GitHub or a package index:** it may not be
    the version the model was trained with.
  - **Importing the thesis's `predict_viz.py` from its folder:**
    - it ties the app to a folder outside the repository;
    - it pulls in plotting code;
    - it loads checkpoints without `weights_only`.
  - **Colours stored as PLY face properties:** some viewers, Blender among
    them, ignore them.

## 23. MeshPilot viewer: its own small app, saved predictions, plain WebGL

- **Chosen:**
  - **App:** `meshpilot.api`, a FastAPI app of its own (an `APIRouter` plus
    `create_app`).
  - **Data:** it serves the predictions `meshpilot.predict` wrote, and the
    test-split context `meshpilot.verify` wrote for the same checkpoint.
    Before serving a mesh, it checks that the mesh and its labels still have
    the hashes recorded at prediction time.
  - **Page:** one static page with about 150 lines of hand-written WebGL. It
    has:
    - layers for the original mesh, the predictions and the dataset labels;
    - a class legend that highlights the faces predicted as that class.
      Every other face is drawn as a translucent ghost, or hidden when
      "Isolate" is on;
    - three preset chat questions per dataset, each naming one of the
      model's classes;
    - a panel with provenance, test-split context, single-mesh scores and
      limitations.
- **Why:**
  - **A separate app:** the RAG app refuses to start without its index and
    API key (entry 17), and it must keep working without any MeshPilot file.
    The viewer needs neither the index nor the key.
  - **Saved predictions:** they keep torch and the checkpoint out of the web
    process. Every prediction shown is a recorded, reproducible run with its
    provenance.
  - **Plain WebGL:** the viewer needs one mesh, per-face colours and an orbit
    camera. Written directly, there is nothing to download, vendor or build,
    and the page loads nothing from the network.
  - **Translucent ghost:** highlighted faces are often inside the part, such
    as the floor of a pocket.
- **Rejected:**
  - **Mounting the viewer in `rag.api`:** the viewer would then depend on the
    RAG index and API key, or that app's fail-fast startup would have to be
    weakened.
  - **three.js, vendored or from a CDN:** a dependency of several hundred
    kilobytes, or a network request on every page load, for features the
    viewer does not use.
  - **Running inference per request:** it puts torch and the weights in the
    server, and takes seconds per mesh.
  - **Hiding the faces that are not highlighted:** without them, a
    highlighted region loses its place on the part.

## 24. MeshPilot chat: one geometry tool, with the highlight decided by code

- **Chosen:**
  - **Loop:** `meshpilot/agent.py` repeats the document agent's loop with
    the same rules (entry 20):
    - one tool per step, at most 5 steps, and tools off on the last step;
    - assistant turns sent back unchanged;
    - a structured final reply.

    It reuses document search and the calculator through a subclass of
    `AgentRun`.
  - **Tool:** `show_predicted_class` takes a class name, which its schema
    limits to the model's own classes. Code then:
    - finds the faces predicted as that class;
    - groups them into patches of faces that share an edge;
    - sends the face ids to the viewer, which highlights them.

    Claude receives only the counts and a caveat.
  - **Evidence:** the reply keeps document citations and model-prediction
    results in separate fields, shown in separate boxes.
  - **Optional parts:** without an API key, the chat is off and the viewer
    still works. Without the RAG index, document search is not offered, and
    the prompt says so.
- **Why:**
  - **A separate loop:** `rag/agent.py` is what the A3 evaluation runs.
    Adding a tool to it would change an evaluated configuration.
  - **Highlight decided by code:** a highlight computed from the predictions
    cannot misquote them. Claude only chooses the class and writes the text.
  - **Counts, not face ids, for Claude:** face ids are long and add nothing
    to the answer.
  - **The caveat on patches:** one feature can be split into several
    patches, and touching features can merge into one.
- **Rejected:**
  - **Adding the tool to `rag/agent.py`:** that changes the evaluated agent.
  - **Letting Claude return face ids or labels in its reply:** it could
    invent them.
  - **Reporting patches as feature counts:** they are not validated.
  - **Patches joined at shared vertices:** two regions touching at one
    corner would merge.

## 25. MeshPilot chat provider: Perplexity's Agent API as an alternative to Claude

- **Chosen:**
  - **The adapter:** `meshpilot/perplexity.py` presents the one client
    method the chat uses, `beta.messages.create`. It translates the request
    and reply to Perplexity's Agent API (`POST /v1/agent`), so
    `meshpilot/agent.py`, the tools and the highlighting are unchanged.
  - **What it sends:** our function tools only, never web search, URL
    fetching, the sandbox or a preset; `max_steps: 1`, `store: false`, and
    a capped `max_output_tokens`.
  - **How it connects:** the standard library's HTTP client, with no
    automatic retries.
  - **Choosing the provider:** `MESHPILOT_CHAT_PROVIDER` picks it. If unset,
    the provider whose key is set is used, Claude first.
  - **Default model:** `openai/gpt-6-luna`, the cheapest OpenAI model in
    Perplexity's list.
  - **Smoke test:** `meshpilot.smoke` runs three scenarios. Before each
    request it checks a request count and a worst-case dollar limit.
  - **Per-question cap:** one viewer question makes at most
    `MESHPILOT_CHAT_MAX_STEPS` requests (default 3, against the document
    agent's 5). That covers one tool call, one more if needed, and the
    answer.
- **Why:**
  - **Spend:** the owner has Perplexity credit and wanted no new Anthropic
    spending.
  - **No hidden costs:** Perplexity's built-in tools are billed per call and
    would search the web. Leaving them out keeps the local documents as the
    only evidence.
  - **Choice of model:** the docs do not list per-model function-calling
    support. Perplexity's own function-calling example uses an OpenAI model,
    and the smoke test's first scenario checks the tool call before
    anything relies on it.
- **Rejected:**
  - **Sonar Chat Completions:** its support ended on 27 September 2026.
  - **The Router (Orchestrator) API:** it is in private preview and offers
    only Perplexity-hosted open models.
  - **Perplexity's or OpenAI's Python SDK:** a new dependency for a single
    POST.
  - **A second copy of the chat loop:** the translation layer keeps one
    loop.
  - **Removing the Anthropic adapter:** both providers stay, and the
    viewer labels which one is active.

## 26. MeshPilot demo corpus: one project document, numbers filled from records, its own index

- **Chosen:**
  - **The document:** the MeshPilot notes, written in this project,
    `docs/meshpilot/demo-corpus/meshpilot-demo-notes.md`. It covers what
    each class means (with pinned links to the dataset's documentation),
    the model's provenance, the test-split scores, one fixed reference mesh,
    the limitations, an example review and the evidence rules.
  - **Its numbers:** the Markdown holds none. Each figure is a placeholder
    filled from the `meshpilot.verify` and `meshpilot.predict` records.
  - **The build** (`python -m meshpilot.corpus build`):
    - checks the records against each other and against the checkpoint,
      split, mesh and labels they describe;
    - recomputes the reference mesh's scores;
    - stops on any unfilled placeholder;
    - renders one PDF page per section, with clickable links;
    - reads the text, the table rows, the links and each chunk's page number
      back from the PDF before indexing it.
  - **The index:** `MP1`, with R1's chunking and `BAAI/bge-small-en-v1.5`
    pinned to revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, through a
    new optional `revision` argument on `rag.embeddings.Embedder`.
  - **Where it lives:** the PDF, its manifest and the index are in
    `data/meshpilot/`, which is gitignored and outside the Docker context.
  - **What uses it:** the viewer's chat searches only this index. The
    geotechnical manifest, R1–R3 and the RAG API are unchanged.
- **Why:**
  - **Evidence the chat can cite:** the geotechnical manuals cannot answer
    questions about these classes, and the corpus rules exclude the thesis
    and other personal documents.
  - **No hand-typed numbers:** values typed in by hand would drift from the
    measured ones.
  - **One section per page:** a citation then names the section it came
    from.
  - **A pinned model revision:** it fixes the weights the index was built
    with.
- **Rejected:**
  - **Adding the notes to the geotechnical corpus:** it would change R1 and
    its evaluation.
  - **Committing the PDF:** it is a build product, and PDFs are never
    committed.
  - **A Markdown loader in `rag`:** PDF pages reuse the existing loader,
    chunker and page-number rules unchanged.
  - **An unpinned model:** a silent upstream change could alter retrieval.
- **What the first retrieval checks found:**
  - **A limitations question** found its page first, clearly ahead of the
    rest.
  - **A class-definition question** found the definitions page only third,
    with nearly tied scores, because the score tables on pages 4 and 5 also
    name the classes.
  - **An unsupported question** (machining speeds) still retrieved the
    nearest pages, with scores above those of the class-definition question.
  - **So the scores do not show whether a question is answerable:** the
    model must judge that from the excerpts.
- **Baseline:** in the owner-approved chat check, the model searched with
  its own wording, retrieved the definitions page first and cited it
  correctly. `MP1` was then frozen as the baseline: chunking, model and
  definitions are not tuned on single queries.

## 27. v1 release packaging: a key-free MOCK demo and published result records

- **Chosen:**
  - **The mock demo:** `python -m meshpilot.mock_demo` serves:
    - a synthetic stepped shaft generated in code;
    - labels from a geometric rule, recorded with `"mock": true`;
    - a scripted chat that calls the real geometry tool.

    The viewer shows "MOCK data", "MOCK chat" and "MOCK: no model ran".
  - **Published records:** `eval/results/meshpilot/` holds copies of the
    local verify and smoke records, with local paths removed and the local
    records' SHA-256 kept. The README quotes numbers only from these.
  - **The README:** it is now a short overview. The detailed guides moved to
    `docs/rag.md` and `docs/meshpilot/usage.md`, with local paths replaced
    by placeholders.
- **Why:**
  - **A demo needs its own data:** a fresh clone has no checkpoint, dataset
    or key, and the Fusion 360 Gallery licence forbids redistributing its
    meshes.
  - **Labels:** the MOCK labels make the demo impossible to mistake for real
    inference.
  - **The README rule:** the owner's rule allows only committed measurements
    in the README.
- **Rejected:**
  - **Shipping a dataset mesh or a real prediction:** the licence forbids
    it, and the assets stay private.
  - **Replaying recorded live responses in the demo:** they could be taken
    for live calls.
  - **Committing the local records unchanged:** they contain private paths.

## 28. Public release: a new repository, and Apache-2.0 for original work only

- **Chosen:**
  - **A new repository with a fresh history:**
    - The public copy is exported from the development repository, without
      its internal working notes.
    - It starts from one reviewed commit.
    - The development history is not published.
  - **Licence:** Apache-2.0.
    - `NOTICE` and the README scope it to the original source code, tests,
      documentation and measurement records here.
    - It does not cover datasets, checkpoints, generated outputs or
      third-party assets.
  - **PyMuPDF stays:**
    - Its AGPL-3.0-or-commercial status is stated in the README and in
      `NOTICE`.
    - No Docker image is published, because the image would include
      PyMuPDF. The `Dockerfile` is for local builds.
- **Why:**
  - **History:** the development history holds working notes with local
    paths, and its commits carry a private e-mail address. A fresh history
    leaves both out without rewriting anything.
  - **Apache-2.0:** permissive, with an explicit patent grant. Code under
    Apache-2.0 may be combined with AGPL-3.0 code, so it can be used
    alongside PyMuPDF.
- **Rejected:**
  - **Making the development repository public:** its history would
    publish the local paths.
  - **Rewriting its history:** it needs a force-push, and a path missed in
    one of its commits would still be published.
  - **An orphan release branch in the same repository:** a later change of
    visibility would expose the rest of the history.
  - **AGPL-3.0 for this code:** using PyMuPDF does not require it, and it
    restricts reuse more than needed.
  - **Replacing PyMuPDF with a permissive PDF library:** it would change
    the frozen MP1 baseline and the PDF pipeline during the feature freeze.
