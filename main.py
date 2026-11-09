from dotenv import load_dotenv

from ghg_extraction.helper.utils import load_config
from ghg_extraction.parsing import PdfParser
from ghg_extraction.indexing import ReportIndexer
from ghg_extraction.retriever import RetrieverModule

def main():
    
    load_dotenv()
    
    # Load configuration
    config = load_config("config.yaml")
    prompts = load_config("prompts.yaml")
    
    # ATTENTION: Initializing and parsing documents takes a long time. 
    # pdf_parser = PdfParser(config)
    # pdf_parser.parse_documents()

    # ATTENTION: Indexing all reports takes a long time.
    # One vector store per report, one chunk per page
    # indexer = ReportIndexer(config)
    # indexer.create_vector_stores()

    # Top-k pages per report for the global search query
    retriever = RetrieverModule(config, prompts)
    retriever.run_retrieval()

    # rag = RAG(config)
    # rag.run_indexing()
    # rag.run_retrieval(enable_cross_ranking=False)
    # rag.run_generation(enable_probs=True)

if __name__ == "__main__":
    main()
