# main.py
import argparse
from graph1_pdf_upload import run_graph1
from graph2 import run_graph2

USAGE = """\
python3 main.py graph1 <pdf_path>
python3 main.py graph2 "<query>" [--doc <doc_id>] [--retries N] [--th THRESHOLD]
"""

def cli_graph1(args):
    doc_id = run_graph1(args.pdf_path)
    print("doc_id:", doc_id)


def cli_graph2(args):
    result = run_graph2(
        args.query,
        doc_id=args.doc,
        max_retries=args.retries,
        threshold=args.th
    )
    print("\nAnswer:\n", result.get("answer"))
    print("\nMetrics:", result.get("metrics"))
    print("\nSources (page, chunk_id, score):")
    for s in result.get("sources", []):
        print(f"  p{s.get('page')}  {s.get('chunk_id')}  {s.get('rerank_score'):.3f}")


def build_parser():
    parser = argparse.ArgumentParser(description="RAG Graph Runner", usage=USAGE)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # graph1
    p1 = sub.add_parser("graph1", help="Ingest a PDF (Graph-1)")
    p1.add_argument("pdf_path", type=str, help="Path to PDF file")
    p1.set_defaults(func=cli_graph1)

    # graph2
    p2 = sub.add_parser("graph2", help="Run RAG QA (Graph-2)")
    p2.add_argument("query", type=str, help="User query in quotes")
    p2.add_argument("--doc", type=str, default=None, help="Optional doc_id to filter retrieval")
    p2.add_argument("--retries", type=int, default=2, help="Max retries if metrics below threshold")
    p2.add_argument("--th", type=float, default=0.7, help="Metrics threshold (0..1)")
    p2.set_defaults(func=cli_graph2)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)