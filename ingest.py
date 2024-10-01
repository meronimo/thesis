import logging

from stages.loading import Loading
from stages.storing import Storing
from utils.config import ROOT_DIR

# initialize logging for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ingestion():
    # load documents
    # for live purpose take: wiki_movie_plots.json
    # for testing purpose take: wiki_movie_plots_1000.json
    d = Loading(input_file=f"{ROOT_DIR}/data/wiki_movie_plots_512_50_movie_2024.json")
    documents = d.get_documents()
    # from llama_index.core import Document
    # documents = [Document(
    #     text="""
    #         Release Year: 2024
    #         Title: Road House
    #         Directed: Doug Liman
    #         Genre: Action, Drama
    #         Cast: Jake Gyllenhaal, Daniela Melchior, Billy Magnussen, Jessica Williams, Joaquim de Almeida, Conor McGregor,
    #         Wikipedia: https://en.wikipedia.org/wiki/Road_House_(2024_film)
    #         Plot:
    #     """,
    #     metadata={"release_year": 2024, "title": "Road House"},
    #     excluded_llm_metadata_keys=["file_name"],
    #     text_template="Metadata:\n{metadata_str}\n-----------\nContent:\n{content}"
    # )]

    # store documents
    db = "qdrant"  # qdrant
    # chromadb resulted in a lot of errors

    # chunk_size, overlap_size
    chunk_sizes_overlap_size = {
        512: 50,
        # 1024: 100,
        # 2048: 200,
    }
    for chunk_size, overlap_size in chunk_sizes_overlap_size.items():
        logger.info(f"Storing documents with chunk_size: {chunk_size} and overlap_size: {overlap_size}")
        s = Storing(
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            collection_name=f"wiki_movie_plots_{chunk_size}_{overlap_size}_mxbai_final"
        )
        s.store_data(documents, db)


if __name__ == '__main__':
    ingestion()
