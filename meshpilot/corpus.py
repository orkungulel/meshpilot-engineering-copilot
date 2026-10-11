"""The MeshPilot demo corpus: one project document, built into a local PDF with its own index.

    python -m meshpilot.corpus build --prediction <id>   check, fill and build the PDF, manifest and index
    python -m meshpilot.corpus search "question"          show what the chat would retrieve (no LLM)

The notes' Markdown source (docs/meshpilot/demo-corpus/) holds no numbers.
Every figure is a {{placeholder}} filled from the meshpilot.verify and
meshpilot.predict records. Before filling anything, the build checks:
- the two records against each other;
- the records against the files they describe: the checkpoint, the split,
  the mesh and its labels;
- the reference mesh's scores, recomputed from the recorded labels.

It stops on any mismatch and on any placeholder left unfilled. Each "## "
section becomes one PDF page, so a citation's page names its section; the
text, tables, links and page numbers are read back from the PDF and checked.

The PDF, its manifest and the index go to data/meshpilot/, which is
gitignored and outside the Docker context. The geotechnical corpus and its
R1-R3 indexes are never touched.
"""

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf

from meshpilot.outputs import OUTPUTS_DIR, read_labels, single_mesh_scores
from rag import config
from rag.api import QAService
from rag.chunking import chunk_pages
from rag.config import IndexConfig
from rag.corpus import sha256_of
from rag.index import VectorIndex, build_corpus_index
from rag.pdf_loader import load_pages

TEMPLATE = config.ROOT / "docs" / "meshpilot" / "demo-corpus" / "meshpilot-demo-notes.md"
CORPUS_DIR = config.DATA_DIR / "meshpilot"
PDF_PATH = CORPUS_DIR / "raw" / "meshpilot-demo-notes.pdf"
MANIFEST_PATH = CORPUS_DIR / "manifest.csv"
DOC_ID = "meshpilot-notes"
TITLE = "MeshPilot demo notes (project document, research prototype)"
# R1's embedding model and chunking, with the model pinned to one Hugging Face commit.
INDEX_CONFIG = IndexConfig("MP1", "BAAI/bge-small-en-v1.5", chunk_words=300, overlap_words=60)
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
INDEX_DIR = CORPUS_DIR / "index" / INDEX_CONFIG.name
SEARCH_RESULTS = 3

PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
CSS = """
* { font-family: sans-serif; font-size: 9.5pt; line-height: 1.35; }
h1 { font-size: 15pt; margin: 0 0 8pt 0; }
h2 { font-size: 12.5pt; margin: 0 0 8pt 0; }
p { margin: 0 0 6pt 0; }
ul, ol { margin: 0 0 6pt 0; padding-left: 14pt; }
li { margin: 0 0 3pt 0; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; }
th, td { border: 0.5pt solid #888888; padding: 2pt 5pt; text-align: left; }
th { background-color: #eeeeee; }
code { font-family: monospace; }
.link { white-space: nowrap; color: #1a4f8a; text-decoration: underline; }
"""


class BuildError(Exception):
    """The records, the template or the PDF failed a check, so nothing was indexed."""


def resolve(path: str) -> Path:
    """A path as a record stores it; relative paths were relative to the repository."""
    p = Path(path)
    return p if p.is_absolute() else config.ROOT / p


# ---------------------------------------------------------------- provenance
def load_records(prediction_path: Path) -> tuple[dict, dict, Path]:
    """The predict record and the verify record of the same checkpoint (in the folder above)."""
    verify_path = prediction_path.parent.parent / "verify.json"
    if not prediction_path.is_file() or not verify_path.is_file():
        raise BuildError(f"need {prediction_path.name} and the checkpoint's verify.json; run predict and verify first")
    return (json.loads(prediction_path.read_text(encoding="utf-8")),
            json.loads(verify_path.read_text(encoding="utf-8")), verify_path)


def check_provenance(prediction: dict, verify: dict) -> dict:
    """Check the records against each other and against the files they describe. Returns the split."""
    problems = []
    model, mesh, truth = prediction["model"], prediction["mesh"], prediction.get("ground_truth")
    if verify.get("agrees") is not True:
        problems.append("meshpilot.verify did not agree with the training run's recorded scores")
    if verify["checkpoint_sha256"] != model["checkpoint_sha256"] or verify["epochs_completed"] != model["epochs_completed"]:
        problems.append("the verify and predict records name different checkpoints")
    checkpoint = resolve(model["checkpoint"])
    if not checkpoint.is_file() or sha256_of(checkpoint) != model["checkpoint_sha256"]:
        problems.append("the checkpoint file is missing or no longer matches its recorded SHA-256")
    split_path = resolve(mesh["split_file"])
    if not split_path.is_file() or sha256_of(split_path) != mesh["split_sha256"]:
        raise BuildError("the split file is missing or no longer matches its recorded SHA-256")
    verify_split = resolve(verify["test_split"]["split_file"])
    if not verify_split.is_file() or sha256_of(verify_split) != mesh["split_sha256"]:
        problems.append("verify and predict used different split files")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if mesh["split"] != "test" or mesh["stem"] not in split["test"] or mesh["stem"] in split["train"]:
        problems.append(f"{mesh['stem']} is not a test-only mesh of the recorded split")
    if len(split["test"]) != verify["test_split"]["meshes"]:
        problems.append("the split's test list and the verified test set differ in size")
    obj = resolve(mesh["obj"])
    if not obj.is_file() or sha256_of(obj) != mesh["obj_sha256"]:
        problems.append("the reference mesh is missing or no longer matches its recorded SHA-256")
    if truth is None:
        problems.append("the reference mesh has no dataset labels, so it has no scores")
    elif not resolve(truth["labels_file"]).is_file() or sha256_of(resolve(truth["labels_file"])) != truth["labels_sha256"]:
        problems.append("the reference mesh's labels are missing or no longer match their recorded SHA-256")
    else:
        recomputed = single_mesh_scores(read_labels(resolve(truth["labels_file"])),
                                        np.asarray(prediction["face_labels"]), model["n_class"])
        recomputed_iou = {prediction["class_names"][c]: v for c, v in recomputed["per_class_iou"].items()}
        recorded = truth["single_mesh_scores"]
        same = (recomputed_iou.keys() == recorded["per_class_iou"].keys()
                and all(abs(v - recorded["per_class_iou"][k]) < 1e-12 for k, v in recomputed_iou.items())
                and abs(recomputed["accuracy"] - recorded["accuracy"]) < 1e-12
                and abs(recomputed["mean_iou"] - recorded["mean_iou"]) < 1e-12)
        if not same:
            problems.append("the reference mesh's recorded scores differ from a recomputation from its labels")
    measured = verify["measured"]
    if abs(float(np.mean(list(measured["per_class_iou"].values()))) - measured["mean_iou"]) > 1e-9:
        problems.append("the verified mean IoU is not the mean of its per-class IoUs")
    if problems:
        raise BuildError("; ".join(problems))
    return split


# ---------------------------------------------------------------- filling the template
def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def fields(prediction: dict, verify: dict, split: dict, hashes: dict) -> dict[str, str]:
    """Every value the template can use, taken from the records; nothing typed in by hand."""
    model, mesh, truth = prediction["model"], prediction["mesh"], prediction["ground_truth"]
    names, test_iou = prediction["class_names"], verify["measured"]["per_class_iou"]
    mesh_scores = truth["single_mesh_scores"]
    lowest = sorted(test_iou, key=test_iou.get)[:4]
    return {
        "input_features": model["input_features"].upper(),
        "width": str(model["C_width"]),
        "blocks": str(model["N_block"]),
        "k_eig": str(model["k_eig"]),
        "n_class": str(model["n_class"]),
        "epochs_completed": str(model["epochs_completed"]),
        "epochs_planned": str(model["epochs_planned"]),
        "selection": ("It was chosen by its test score, so its test scores are optimistic."
                      if model["chosen_by_test_miou"] else "It was not chosen by its test score."),
        "checkpoint_sha": model["checkpoint_sha256"][:12],
        "train_meshes": f"{len(split['train']):,}",
        "test_meshes": f"{len(split['test']):,}",
        "split_sha": mesh["split_sha256"][:12],
        "verify_record_sha": hashes["verify"][:12],
        "prediction_record_sha": hashes["prediction"][:12],
        "score_tolerance": f"{verify['tolerances']['scores']:g}",
        "test_miou": f"{verify['measured']['mean_iou']:.4f}",
        "recorded_miou": f"{verify['recorded_in_history']['mean_iou']:.4f}",
        "test_accuracy": f"{verify['measured']['accuracy']:.4f}",
        "recorded_accuracy": f"{verify['recorded_in_history']['accuracy']:.4f}",
        "test_split_iou_table": table(["Class", "Test-split IoU"], [[n, f"{test_iou[n]:.3f}"] for n in names]),
        "lowest_iou_classes": ", ".join(lowest[:-1]) + " and " + lowest[-1],
        "mesh_id": mesh["stem"],
        "mesh_triangles": f"{mesh['faces']:,}",
        "mesh_sha": mesh["obj_sha256"][:12],
        "mesh_class_table": table(
            ["Class", "Predicted triangles", "Labelled triangles", "IoU on this mesh"],
            [[n, f"{prediction['predicted_face_counts'][n]:,}", f"{truth['face_counts'][n]:,}",
              f"{mesh_scores['per_class_iou'][n]:.3f}" if n in mesh_scores["per_class_iou"] else "not counted"]
             for n in names]),
        "mesh_accuracy": f"{mesh_scores['accuracy']:.3f}",
        "mesh_miou": f"{mesh_scores['mean_iou']:.3f}",
        "mesh_classes_present": str(len(mesh_scores["per_class_iou"])),
        "mesh_classes_absent": ", ".join(n for n in names if n not in mesh_scores["per_class_iou"]) or "none",
    }


def fill(template: str, values: dict[str, str]) -> str:
    """The template with every {{name}} replaced. Any placeholder without a value stops the build."""
    missing = sorted({name for name in PLACEHOLDER.findall(template) if name not in values})
    if missing:
        raise BuildError(f"unresolved placeholders: {', '.join(missing)}")
    filled = PLACEHOLDER.sub(lambda m: values[m.group(1)], template)
    if "{{" in filled or "}}" in filled:
        raise BuildError("the filled document still contains '{{' or '}}'")
    return filled


# ---------------------------------------------------------------- Markdown to PDF
def sections(markdown: str) -> list[tuple[str, str]]:
    """(heading, Markdown) for each PDF page: the title block first, then one per "## " section."""
    parts = [part.strip() for part in re.split(r"(?m)^(?=## )", markdown) if part.strip()]
    return [(part.splitlines()[0].lstrip("#").strip(), part) for part in parts]


def inline(text: str, links: list) -> str:
    """Escaped text with links, bold, code and italics; each link is recorded for its annotation."""
    def link(match):
        links.append((match.group(1), match.group(2)))
        return f'<span class="link">{html.escape(match.group(1))}</span>'

    parts, last = [], 0
    for match in LINK.finditer(text):
        parts.append(html.escape(text[last:match.start()], quote=False))
        parts.append(link(match))
        last = match.end()
    parts.append(html.escape(text[last:], quote=False))
    out = "".join(parts)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"<i>\1</i>", out)


def to_html(markdown: str, links: list) -> str:
    """The Markdown this document uses (headings, paragraphs, two-level lists, tables) as HTML."""
    out, paragraph, items = [], [], []  # items: [level, ordered, text]

    def flush():
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph), links)}</p>")
            paragraph.clear()
        if items:
            html_list, open_tags = [], []
            for level, ordered, text in items:
                while len(open_tags) > level + 1:
                    html_list.append(f"</li></{open_tags.pop()}>")
                if len(open_tags) == level + 1:
                    html_list.append("</li>")
                else:
                    tag = "ol" if ordered else "ul"
                    open_tags.append(tag)
                    html_list.append(f"<{tag}>")
                html_list.append(f"<li>{inline(text, links)}")
            while open_tags:
                html_list.append(f"</li></{open_tags.pop()}>")
            out.append("".join(html_list))
            items.clear()

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line, stripped = lines[i], lines[i].strip()
        item = re.match(r"^(\s*)(-|\d+\.)\s+(.*)$", line)
        if stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
                i += 1
            head = "".join(f"<th>{inline(c, links)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{inline(c, links)}</td>" for c in row) + "</tr>" for row in rows[2:])
            out.append(f"<table><tr>{head}</tr>{body}</table>")
            continue
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{inline(stripped.lstrip('#').strip(), links)}</h{level}>")
        elif item:
            if paragraph:
                flush()
            items.append([0 if len(item.group(1)) < 2 else 1, item.group(2) != "-", item.group(3)])
        elif not stripped:
            flush()
        elif items and line.startswith(" "):
            items[-1][2] += " " + stripped  # a wrapped list item
        else:
            if items:
                flush()
            paragraph.append(stripped)
        i += 1
    flush()
    return "".join(out)


def build_pdf(markdown: str, path: Path) -> list[dict]:
    """One A4 page per section, with clickable links. Returns each page's heading and links."""
    pdf = pymupdf.open()
    layout = []
    for heading, part in sections(markdown):
        links: list[tuple[str, str]] = []
        page = pdf.new_page(width=595, height=842)
        spare, _ = page.insert_htmlbox(page.rect + (50, 50, -50, -50), to_html(part, links), css=CSS, scale_low=1)
        if spare < 0:
            raise BuildError(f"the section {heading!r} does not fit on one page")
        for text, uri in links:
            hits = page.search_for(text)
            if not hits:
                raise BuildError(f"the link text {text!r} is not on the page of {heading!r}")
            for rect in hits:
                page.insert_link({"kind": pymupdf.LINK_URI, "from": rect, "uri": uri})
        layout.append({"heading": heading, "links": links})
    pdf.set_metadata({"title": TITLE, "author": "MeshPilot project",
                      "subject": "Research prototype notes; not a standard, procedure or compliance check",
                      "creator": "meshpilot.corpus"})
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(path, garbage=3, deflate=True)
    pdf.close()
    return layout


def flat(text: str) -> str:
    return " ".join(text.split())


def check_pdf(path: Path, layout: list[dict], prediction: dict, verify: dict) -> dict:
    """Read the PDF back the way indexing reads it and check its pages, text, tables, links and chunks."""
    pages = load_pages(path, DOC_ID)
    problems = []
    if len(pages) != len(layout):
        problems.append(f"{len(pages)} pages, expected {len(layout)}")
    for page, expected in zip(pages, layout):
        if flat(expected["heading"]) not in flat(page.text):
            problems.append(f"page {page.number}: heading {expected['heading']!r} not extracted")
        if "{{" in page.text or "}}" in page.text:
            problems.append(f"page {page.number}: an unfilled placeholder")
    by_heading = {expected["heading"].split(".")[0]: page for page, expected in zip(pages, layout)}
    names, test_iou = prediction["class_names"], verify["measured"]["per_class_iou"]
    truth = prediction["ground_truth"]
    mesh_iou = truth["single_mesh_scores"]["per_class_iou"]
    test_page, mesh_page = flat(by_heading["3"].text), flat(by_heading["4"].text)
    for n in names:
        if f"{n} {test_iou[n]:.3f}" not in test_page:
            problems.append(f"test-split table row for {n} not extracted as expected")
        cells = [n, f"{prediction['predicted_face_counts'][n]:,}", f"{truth['face_counts'][n]:,}",
                 f"{mesh_iou[n]:.3f}" if n in mesh_iou else "not counted"]
        if " ".join(cells) not in mesh_page:
            problems.append(f"reference-mesh table row for {n} not extracted as expected")
    with pymupdf.open(path) as pdf:
        for page, expected in zip(pdf, layout):
            uris = {link.get("uri") for link in page.get_links()}
            for _, uri in expected["links"]:
                if uri not in uris:
                    problems.append(f"page {page.number + 1}: link to {uri} is not clickable")
    chunks = chunk_pages(pages, INDEX_CONFIG.chunk_words, INDEX_CONFIG.overlap_words)
    for page, expected in zip(pages, layout):
        opening = flat(expected["heading"])[:40]
        if not any(c.page == page.number and opening in c.text for c in chunks):
            problems.append(f"no chunk on page {page.number} carries its heading")
    if problems:
        raise BuildError("; ".join(problems))
    return {"pages": [(p.number, e["heading"]) for p, e in zip(pages, layout)], "chunks": len(chunks),
            "links": sum(len(e["links"]) for e in layout)}


# ---------------------------------------------------------------- manifest and index
def write_manifest(pdf_path: Path, page_count: int) -> None:
    row = {"doc_id": DOC_ID, "title": TITLE, "publisher": "MeshPilot project", "year": "2026",
           "publication_number": "", "url": "docs/meshpilot/demo-corpus/meshpilot-demo-notes.md",
           "licence": "project document", "sha256": sha256_of(pdf_path), "pages": str(page_count)}
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def build_index(pdf_path: Path, provenance: dict, embedder=None, index_dir: Path = INDEX_DIR) -> dict:
    if embedder is None:
        from rag.embeddings import Embedder

        embedder = Embedder(INDEX_CONFIG.embedding_model, revision=EMBEDDING_REVISION)
    index, info = build_corpus_index({DOC_ID: pdf_path}, INDEX_CONFIG, embedder)
    info["embedding_revision"] = getattr(embedder, "revision", None)
    info["provenance"] = provenance
    index.save(index_dir, info)
    return info


def load_search(embedder=None, index_dir: Path = INDEX_DIR):
    """The demo corpus's search and titles, or (None, {}) when it has not been built."""
    if not (index_dir / "index.faiss").exists():
        return None, {}
    info = json.loads((index_dir / "build_info.json").read_text(encoding="utf-8"))
    if embedder is None:
        from rag.embeddings import Embedder

        embedder = Embedder(INDEX_CONFIG.embedding_model, revision=info.get("embedding_revision"))
    titles = {DOC_ID: TITLE}
    index = VectorIndex.load(index_dir, embedder)
    # QAService.search runs one search at a time: the tokenizer is not thread-safe.
    return QAService(index, None, "", titles, INDEX_CONFIG.name, k=SEARCH_RESULTS).search, titles


# ---------------------------------------------------------------- commands
def build_command(args) -> int:
    prediction_path = args.outputs / args.prediction / "prediction.json"
    try:
        prediction, verify, verify_path = load_records(prediction_path)
        split = check_provenance(prediction, verify)
        hashes = {"verify": sha256_of(verify_path), "prediction": sha256_of(prediction_path),
                  "template": sha256_of(TEMPLATE)}
        markdown = fill(TEMPLATE.read_text(encoding="utf-8"), fields(prediction, verify, split, hashes))
        layout = build_pdf(markdown, PDF_PATH)
        try:
            report = check_pdf(PDF_PATH, layout, prediction, verify)
        except BuildError:
            PDF_PATH.unlink(missing_ok=True)  # never leave a PDF that failed its checks
            raise
    except BuildError as exc:
        print(f"build stopped: {exc}")
        return 1
    write_manifest(PDF_PATH, len(report["pages"]))
    print("provenance: records agree with each other, the checkpoint, the split, the mesh and its labels")
    print(f"PDF: {len(report['pages'])} pages, {report['links']} clickable links, {report['chunks']} chunks, "
          "every heading and table row read back on its page")
    for number, heading in report["pages"]:
        print(f"  page {number}: {heading}")
    if args.no_index:
        return 0
    info = build_index(PDF_PATH, {"records_sha256": hashes, "pdf_sha256": sha256_of(PDF_PATH)})
    print(f"index {INDEX_CONFIG.name}: {info['chunks']} chunks from {info['pages']} pages, "
          f"{info['embedding_model']} at revision {info['embedding_revision']}, in {INDEX_DIR}")
    return 0


def search_command(args) -> int:
    search, titles = load_search()
    if search is None:
        print("the demo corpus has not been built: run python -m meshpilot.corpus build first")
        return 2
    for rank, hit in enumerate(search(args.question)[: args.k], 1):
        excerpt = flat(hit.chunk.text)
        print(f"{rank}. score {hit.score:.3f} | {titles.get(hit.chunk.doc_id, hit.chunk.doc_id)}, page {hit.chunk.page}")
        print(f"   {excerpt[:420]}{'...' if len(excerpt) > 420 else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or search the MeshPilot demo corpus.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="check the records, fill the notes, build the PDF, manifest and index")
    build.add_argument("--prediction", required=True, help="prediction id, as GET /predictions lists it")
    build.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    build.add_argument("--no-index", action="store_true", help="build and check the PDF only")
    search = commands.add_parser("search", help="show the passages the chat would retrieve (no LLM)")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=SEARCH_RESULTS)
    args = parser.parse_args(argv)
    return build_command(args) if args.command == "build" else search_command(args)


if __name__ == "__main__":
    sys.exit(main())
