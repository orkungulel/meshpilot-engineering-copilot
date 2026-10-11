# Document QA (RAG): setup, corpus, API, Docker and evaluation

The detailed guide for the document QA part. The project overview is in the [README](../README.md).

Work in progress. A question-answering service over a small set of public
geotechnical engineering documents, with retrieval evaluation and a minimal
tool-using agent. Design decisions, with the alternatives I rejected, are in
[DECISIONS.md](../DECISIONS.md).

No results are reported yet. Numbers will appear here only after
`eval/run_eval.py` has measured them, each linked to the run that produced it.

## Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate          # Windows: py -3.11 -m venv .venv, then .venv\Scripts\activate
pip install -e ".[dev]"
pytest
cp .env.example .env          # then put your API key in .env (gitignored, never committed)
```

## Corpus

The PDFs are not in the repository. `corpus/manifest.csv` lists each document
with its source URL and licence and, once pinned, its SHA-256 and page count.

```bash
python -m rag.cli download        # fetch the PDFs into data/raw/ and verify them
python -m rag.cli download --pin  # first download only: record hashes and page counts
python -m rag.cli check           # pages per document, and how many have little or no text
```

The publishers' servers refuse automated downloads from cloud machines (HTTP
403 from GitHub's runners), so on a machine where `download` is refused, save
the PDFs by hand: `python -m rag.cli sources` prints each target file name next
to its URL. `download --pin` then verifies and pins the files already in
`data/raw/` without fetching them again.

## Index and search

```bash
pip install -e ".[embed]"                    # the local embedding model (pulls in torch)
python -m rag.cli build-index --config R1    # R1 is the default; R2 and R3 are the comparisons
python -m rag.cli search "allowable bearing pressure on rock" --config R1
python -m rag.cli ask "What is the allowable bearing pressure on rock?"   # needs ANTHROPIC_API_KEY
python -m rag.cli ask --agent "..."   # the agent, printing each tool call
```

On Linux without a GPU, running
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`
before the `[embed]` install skips the gigabytes of CUDA libraries that the
default torch package brings; the Dockerfile does the same. The first
`build-index` downloads the embedding model from Hugging Face.

Each index lands in `data/index/<config>/` with a `build_info.json` recording
the configuration, the SHA-256 of every document it was built from, the model,
library versions and how many chunks exceeded the embedding model's token limit.

## API

```bash
uvicorn rag.api:app --port 8000      # needs an index (build-index) and ANTHROPIC_API_KEY
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "..."}'
```

`POST /ask` takes `{"question": "...", "mode": "agent"}` (a question of 1 to
2,000 characters; `mode` is `agent`, the default, or `rag`) and returns:

- `answer`: the answer text, or "The retrieved documents do not contain an
  answer I can cite." when the documents do not contain it
- `answerable`: false in that case
- `citations`: one `{doc_id, title, page}` per cited page (1-based PDF page)
- `tool_calls`: the agent's tool calls in order, each `{step, tool, input,
  output_summary}`; empty in `rag` mode
- `model`: the model that wrote the answer

In `agent` mode Claude decides when to use two tools: `search_documents(query)`,
which returns the 5 most similar excerpts, and `calculator(expression)`, which
does arithmetic only. Each step is one model call with at most one tool call;
there are at most 5 steps, and on the last one the tools are switched off, so
the model has to answer from what it has found. Every tool call is logged
(logger `rag.agent`) and returned. `rag` mode is one search and one answer.

An empty or overlong question gets 422; a model that gives no usable answer,
or a failed call to the Anthropic API, gets 502. `GET /health` reports the
served index (`RAG_INDEX`, default R1) and model. The API refuses to start
without an index or an API key.

## Docker

```bash
python -m rag.cli build-index --config R1                       # on the host, once
docker build -t rag .
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/data:/app/data:ro" rag
```

The image holds the code (including `meshpilot`, but no checkpoint, mesh or
prediction), the corpus manifest and the embedding model (so it starts without
reaching Hugging Face). The index is mounted read-only from
`data/`, and the API key comes from `--env-file`; neither is part of the image.
Docker was not available where this was developed, so the image is built and
smoke-tested by CI (`.github/workflows/docker.yml`) whenever the Dockerfile or
the dependencies change.

**Build the image for local use only.**
- It contains PyMuPDF, which is AGPL-3.0 or commercial, so this project
  publishes no image.
- Before distributing an image yourself, review PyMuPDF's licence (see the
  README, "Licences and third-party assets").

## Evaluation

Retrieval is measured on `eval/questions.jsonl`: 30 questions, each with the
document and page (1-based PDF page) that answers it, a reference answer, and
who wrote it. One question per line:

```json
{"id": "q01", "question": "...", "doc_id": "fhwa-gec07", "page": 57, "reference_answer": "...", "type": "number", "written_by": "human"}
```

`type` is one of `fact`, `number`, `definition`, `procedure`, `calculation`;
`written_by` is `human` or `claude-drafted-human-verified`. An optional
`also_acceptable` list (`[{"doc_id": "...", "page": 12}]`) is for a fact the
corpus states on two pages.

```bash
python eval/run_eval.py validate                # structure; plus page-text checks if the PDFs are present
python eval/run_eval.py retrieval --config R1   # also R2 and R3
python eval/run_eval.py report                  # markdown table of all runs
python eval/run_eval.py compare eval/results/<run A> eval/results/<run B>
```

- **recall@k** is the share of questions where a chunk from the expected page
  is among the top k retrieved chunks. With one expected page per question this
  is the same as the hit rate at k. The headline is k = 5; k = 1 shows ranking
  quality, k = 20 the ceiling a reranker could reach, and document-level
  recall@5 separates "wrong document" from "right document, wrong page".
- Every rate comes with a 95% Wilson interval. With 30 questions, differences
  of less than roughly 15 to 20 percentage points are within noise. Two
  configurations are compared with an exact McNemar test on the questions where
  they disagree.
- Each run folder in `eval/results/` records the git commit, the SHA-256 of the
  question file and the index's build record, so every number can be traced
  to the exact code, questions and corpus that produced it.

### Answers

```bash
python eval/run_eval.py judge-check                             # does the judge grade known answers correctly?
python eval/run_eval.py answers --config R1 --limit 5 --reps 1  # pilot: first 5 questions, one run
python eval/run_eval.py answers --config R1                     # full run: every question 3 times
python eval/run_eval.py answers --mode closed-book              # baseline A0: no retrieval
python eval/run_eval.py answers --mode agent --config R1        # A3: the tool-using agent
```

These need `ANTHROPIC_API_KEY`. `eval/unanswerable.jsonl` holds in-domain
questions the corpus does not answer, one per line:
`{"id": "u01", "question": "...", "written_by": "human"}`.

**How answers are judged.** Claude Opus 5.5 (`RAG_JUDGE_MODEL`), a different
model from the one that writes the answers, compares each answer with the
reference answer written by an engineer. An answer is correct only if it gives
the key facts of the reference with matching numbers (a correct unit
conversion is fine), contradicts nothing in it, and actually answers; saying
the documents do not contain the answer counts as incorrect. The judge sees
the question, the reference answer and the system's answer, not the retrieved
text, so it grades correctness rather than faithfulness to the sources.
Before its numbers are used, `judge-check` must show that the judge accepts the
reference answers and rejects an empty answer, "I don't know" and another
question's reference answer.

**Checking the judge against an engineer.** One full run is also graded by
hand, without seeing the judge's verdicts:

```bash
python eval/run_eval.py label-sheet eval/results/<run>   # writes labels.md into the run folder
python eval/run_eval.py agreement eval/results/<run>     # your grades against the judge's
```

`labels.md` starts with the judge's rubric. For each question it shows the
question, the reference answer and the system's answer from the first run that
gave one (`--all-reps` for every run), followed by an empty `label (y/n):`
line. It never shows the judge's verdict, and it is filled in before the run's
`review.md` is opened. Answers where the system said the documents do not
contain the answer are left out, because the rubric makes them incorrect
whoever grades them.

`agreement` reports:

- how often the two agree, with its 95% interval;
- Cohen's kappa, which is agreement beyond what chance would give. This
  matters when most answers are correct, because a judge that passed
  everything would still agree on all of those;
- whether the judge is more lenient or stricter than the engineer;
- every disagreement, with the judge's reason.

The first result is kept in `agreement.json`, because a label changed after
reading the judge's reasons is no longer blind. The agreement will be
reported here.

**Metrics.** Each question is answered 3 times because the model's answers
vary between runs; a question counts as correct when it is answered correctly
in the majority of its runs, and every run's count is shown as well.

- **Correct**: answerable questions answered correctly.
- **Wrongly said "not in the documents"**: answerable questions where the
  system abstained.
- **Abstained on unanswerable**: questions from `eval/unanswerable.jsonl` where
  the system said the documents do not contain the answer instead of guessing.
- **Cited the expected page**: answers whose citations include the question's
  expected page.

Answers served by a model other than the one requested (after a server-side
fallback), refusals, truncated replies and API errors are listed in the run's
`errors.jsonl` and never scored. Each answer run also writes `review.md`, with
every answer next to its reference answer and the judge's verdict.

No results yet: the questions are written once the corpus is downloaded.
