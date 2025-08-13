# main.py
import argparse  # Komut satırı argümanlarını işlemek için
from graph1_pdf_upload import run_graph1  # PDF yükleme ve işleme fonksiyonunu içe aktarır
from graph2 import run_graph2  # RAG QA fonksiyonunu içe aktarır

# Komut satırı kullanım bilgisini tanımlar
USAGE = """\
python3 main.py graph1 <pdf_path>
python3 main.py graph2 "<query>" [--doc <doc_id>] [--retries N] [--th THRESHOLD]
"""

def cli_graph1(args):
    # PDF dosyasını işler ve doc_id döndürür
    doc_id = run_graph1(args.pdf_path)
    print("doc_id:", doc_id)


def cli_graph2(args):
    # Kullanıcı sorgusunu ve diğer parametreleri alarak RAG QA sürecini başlatır
    result = run_graph2(
        args.query,
        doc_id=args.doc,
        max_retries=args.retries,
        threshold=args.th
    )
    print("\nAnswer:\n", result.get("answer"))  # Yanıtı ekrana yazdırır
    print("\nMetrics:", result.get("metrics"))  # Değerlendirme metriklerini yazdırır
    print("\nSources (page, chunk_id, score):")  # Kaynak chunk'ların başlıklarını yazdırır
    # Her bir kaynak chunk'ın sayfa, chunk_id ve skorunu ekrana yazdırır
    for s in result.get("sources", []):
        print(f"  p{s.get('page')}  {s.get('chunk_id')}  {s.get('rerank_score'):.3f}")


def build_parser():
    # Komut satırı argümanlarını ayrıştıracak parser'ı oluşturur
    parser = argparse.ArgumentParser(description="RAG Graph Runner", usage=USAGE)
    sub = parser.add_subparsers(dest="cmd", required=True)  # Alt komutları tanımlar

    # graph1 için alt komut: PDF yükleme
    p1 = sub.add_parser("graph1", help="Ingest a PDF (Graph-1)")
    p1.add_argument("pdf_path", type=str, help="Path to PDF file")  # PDF yolu argümanı
    p1.set_defaults(func=cli_graph1)  # graph1 çağırıldığında çalışacak fonksiyonu ayarlar

    # graph2 için alt komut: RAG QA
    p2 = sub.add_parser("graph2", help="Run RAG QA (Graph-2)")
    p2.add_argument("query", type=str, help="User query in quotes")  # Kullanıcı sorgusu
    p2.add_argument("--doc", type=str, default=None, help="Optional doc_id to filter retrieval")  # Opsiyonel doc_id
    p2.add_argument("--retries", type=int, default=2, help="Max retries if metrics below threshold")  # Maksimum tekrar sayısı
    p2.add_argument("--th", type=float, default=0.7, help="Metrics threshold (0..1)")  # Metrik eşik değeri
    p2.set_defaults(func=cli_graph2)  # graph2 çağırıldığında çalışacak fonksiyonu ayarlar

    return parser


if __name__ == "__main__":
    # Komut satırı argümanlarını ayrıştırır ve ilgili fonksiyonu çalıştırır
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)