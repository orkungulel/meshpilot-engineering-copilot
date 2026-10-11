"""Command-line entry points: python -m rag.cli <command>."""

import argparse
import json
import statistics
import sys
from pathlib import Path

from rag import config
from rag.agent import run_agent
from rag.corpus import download, pin, read_manifest, verify, write_manifest
from rag.embeddings import Embedder
from rag.generate import generate_answer
from rag.index import VectorIndex, build_corpus_index
from rag.llm import make_client
from rag.pdf_loader import load_pages

# A page with fewer words than this is counted as having (almost) no text layer.
MIN_WORDS_PER_PAGE = 20


def cmd_download(args: argparse.Namespace) -> int:
    docs = read_manifest(args.manifest)
    failures = 0
    for i, doc in enumerate(docs):
        try:
            path = download(doc, args.raw_dir)
        except Exception as exc:  # report every failing document, not just the first
            print(f"FAILED    {doc.doc_id}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if doc.sha256:
            status = "verified"
        elif args.pin:
            docs[i] = pin(doc, path)
            status = "pinned"
        else:
            status = "unpinned (check the file, then rerun with --pin)"
        print(f"{status:9} {doc.doc_id}")
    if args.pin:
        write_manifest(args.manifest, docs)
    return 1 if failures else 0


def cmd_sources(args: argparse.Namespace) -> int:
    """Where each document should be saved, and where to get it, for downloading by hand."""
    for doc in read_manifest(args.manifest):
        print(f"{args.raw_dir / (doc.doc_id + '.pdf')}  {doc.url}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    print(f"{'doc_id':30} {'pages':>5} {'<20 words':>9} {'median words/page':>17}")
    for doc in read_manifest(args.manifest):
        path = args.raw_dir / f"{doc.doc_id}.pdf"
        if not path.exists():
            print(f"{doc.doc_id:30} not downloaded")
            continue
        words = [len(page.text.split()) for page in load_pages(path, doc.doc_id)]
        sparse = sum(1 for n in words if n < MIN_WORDS_PER_PAGE)
        flag = "  <- mostly no text layer (scanned?)" if sparse > len(words) / 2 else ""
        print(f"{doc.doc_id:30} {len(words):5} {sparse:9} {statistics.median(words):17.0f}{flag}")
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    index_config = config.INDEX_CONFIGS[args.config]
    pdfs = {}
    for doc in read_manifest(args.manifest):
        path = args.raw_dir / f"{doc.doc_id}.pdf"
        if not path.exists():
            print(f"{doc.doc_id} is not downloaded; run `python -m rag.cli download` first", file=sys.stderr)
            return 1
        verify(doc, path)  # the index must be built from exactly the pinned corpus
        pdfs[doc.doc_id] = path
    index, info = build_corpus_index(pdfs, index_config, Embedder(index_config.embedding_model))
    index.save(args.index_dir / index_config.name, info)
    print(
        f"{index_config.name}: {info['chunks']} chunks from {info['pages']} pages; "
        f"{info['chunks_over_token_limit']} longer than the model's {info['model_max_tokens']}-token limit"
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index_config = config.INDEX_CONFIGS[args.config]
    index = VectorIndex.load(args.index_dir / index_config.name, Embedder(index_config.embedding_model))
    for hit in index.search(args.query, args.k):
        print(f"{hit.score:.3f}  {hit.chunk.doc_id} p.{hit.chunk.page}  {hit.chunk.text[:100]}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    index_config = config.INDEX_CONFIGS[args.config]
    index = VectorIndex.load(args.index_dir / index_config.name, Embedder(index_config.embedding_model))
    titles = {doc.doc_id: doc.title for doc in read_manifest(args.manifest)}
    if args.agent:
        result = run_agent(make_client(), args.model, args.question, lambda query: index.search(query, args.k), titles)
        for call in result.tool_calls:
            print(f"[step {call.step}] {call.tool} {json.dumps(call.input)} -> {call.output_summary}")
        answer = result.answer
    else:
        answer = generate_answer(make_client(), args.model, args.question, index.search(args.question, args.k), titles)
    print(answer.text)
    for citation in answer.citations:
        print(f"  - {citation.title}, page {citation.page}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.cli")
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    download_cmd = commands.add_parser("download", help="download and verify the corpus PDFs")
    download_cmd.add_argument(
        "--pin", action="store_true", help="record SHA-256 and page count of documents not pinned yet"
    )
    download_cmd.set_defaults(func=cmd_download)

    sources_cmd = commands.add_parser("sources", help="list each document's file name and URL, for downloading by hand")
    sources_cmd.set_defaults(func=cmd_sources)

    check_cmd = commands.add_parser("check", help="report page count and text-layer coverage per document")
    check_cmd.set_defaults(func=cmd_check)

    build_cmd = commands.add_parser("build-index", help="chunk and embed the corpus into a FAISS index")
    build_cmd.add_argument("--config", choices=config.INDEX_CONFIGS, default=config.DEFAULT_INDEX)
    build_cmd.set_defaults(func=cmd_build_index)

    search_cmd = commands.add_parser("search", help="show the chunks most similar to a query")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--config", choices=config.INDEX_CONFIGS, default=config.DEFAULT_INDEX)
    search_cmd.add_argument("-k", type=int, default=5)
    search_cmd.set_defaults(func=cmd_search)

    ask_cmd = commands.add_parser("ask", help="answer a question from the corpus, with citations")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--config", choices=config.INDEX_CONFIGS, default=config.DEFAULT_INDEX)
    ask_cmd.add_argument("-k", type=int, default=5, help="chunks given to the model")
    ask_cmd.add_argument("--model", default=config.ANSWER_MODEL)
    ask_cmd.add_argument("--agent", action="store_true", help="let the model search and calculate with tools")
    ask_cmd.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
