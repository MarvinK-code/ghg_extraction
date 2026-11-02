from dotenv import load_dotenv

from ghg_extraction.helper.utils import load_config
from ghg_extraction.parsing import PdfParser
from ghg_extraction.indexing import ReportIndexer

def main():
    
    load_dotenv()
    
    # Load configuration
    config = load_config("config.yaml")
    
    # ATTENTION: Initializing and parsing documents takes a long time. 
    # Uncomment the following lines ONLY if entire document parsing is intended.
    # Once parsed, re-running the following code will parse the raw PDFs again.
    
    ##### --- USE WITH CAUTION --- #####
    
    pdf_parser = PdfParser(config)
    pdf_parser.parse_documents()

    # One vector store per report, one chunk per page
    indexer = ReportIndexer(config)
    indexer.create_vector_stores()

    # rag = RAG(config)
    # rag.run_indexing()
    # rag.run_retrieval(enable_cross_ranking=False)
    # rag.run_generation(enable_probs=True)

if __name__ == "__main__":
    main()
