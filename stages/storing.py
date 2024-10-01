import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import qdrant_client

from utils.config import (
    DEBUG,
    INDEX_PERSIST_DIRECTORY,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    CACHE_DIR,
    DEVICE,
    INDEX_COLLECTION_NAME,
    ROOT_DIR,
    VECTOR_STORE_HOST,
    VECTOR_STORE_PORT
)


class Storing:
    def __init__(
            self,
            chunk_size: int = CHUNK_SIZE,
            overlap_size: int = OVERLAP_SIZE,
            node_parser: bool = False,
            collection_name: str = INDEX_COLLECTION_NAME):
        self.debug = DEBUG
        self.async_mode = True
        self.documents = None
        self.nodes = None
        self.db = None
        self.llm = None
        self.testing = False
        self.vector_store = None
        self.device = DEVICE
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.em_cache_dir = CACHE_DIR
        self.collection_name = collection_name
        self.local_persist_directory = ROOT_DIR + "/data/storage"
        self.index_persist_directory = INDEX_PERSIST_DIRECTORY
        self.node_parser = node_parser or SentenceSplitter(
            chunk_size=int(self.chunk_size),
            chunk_overlap=int(self.overlap_size)
        )
        self.embed_model = HuggingFaceEmbedding(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            cache_folder=self.em_cache_dir,
            device=self.device,
            embed_batch_size=128
        )

    def store_data(self, documents, db, testing=False, llm=None):

        # set the documents and db
        self.documents = documents
        self.db = db
        self.testing = testing
        if self.testing:
            self.llm = self.__set_llm(llm)

        # set vector_store based on db parameter
        self.__set_vector_store()

        # store the nodes in the vector store
        self.__store_nodes(documents)

    def __set_vector_store(self):
        """
        Set the vector store
        :return:
        """
        if self.db is None:
            raise ValueError("Database not set")

        if self.db == "local":
            return True
        if self.db == "chromadb":
            self.__create_chroma_vector_store()
        elif self.db == "qdrant":
            self.__create_qdrant_vector_store()
        else:
            raise ValueError(f"Database {self.db} not supported when creating vector store")

    def __create_chroma_vector_store(self):
        """
        Create a Chroma vector store
        :return:
        """
        # self.__clean_persist_dir() 
        client = chromadb.PersistentClient(path=self.index_persist_directory)
        client.reset()  # instead of cleaning the directory, reset the client
        chroma_collection = client.create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection,
            path=self.index_persist_directory
        )

    def __create_qdrant_vector_store(self):
        """
        Create a Qdrant vector store
        :return:
        """
        llama_debug = LlamaDebugHandler(print_trace_on_end=DEBUG)
        callback_manager = CallbackManager(handlers=[llama_debug])
        client = qdrant_client.QdrantClient(
            host=VECTOR_STORE_HOST,
            port=VECTOR_STORE_PORT
        )
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            callback_manager=callback_manager
        )

    def __store_nodes(self, documents):
        """
        Store the nodes in the vector store
        :param documents:
        :return:
        """
        if self.db == "local":
            self.__store_nodes_local_vector_store()
        else:
            # ingest the documents
            pipeline = IngestionPipeline(
                transformations=[
                    self.node_parser,
                    # self.embed_model
                ],
                documents=self.documents,
                vector_store=self.vector_store
            )

            # run the pipeline to get the nodes
            self.nodes = pipeline.run(
                documents=documents,
                show_progress=self.debug,
            )

        if self.db == "chromadb":
            self.__store_nodes_chroma_vector_store()
        elif self.db == "qdrant":
            self.__store_nodes_qdrant_vector_store()
        else:
            raise ValueError(f"Database {self.db} not supported when storing nodes in vector store")

    def __store_nodes_local_vector_store(self):
        index = VectorStoreIndex.from_documents(
            documents=self.documents,
            show_progress=self.debug,
        )
        # save index to disk
        index.set_index_id("vector_index")
        index.storage_context.persist(self.local_persist_directory)

    def __store_nodes_chroma_vector_store(self):
        """
        Store the nodes in chroma vector store
        """
        # set the storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex(
            use_async=self.async_mode,
            embed_model=self.embed_model,
            storage_context=storage_context,
            nodes=self.nodes,
            show_progress=self.debug
        )

    def __store_nodes_qdrant_vector_store(self):
        """
        Store the nodes in qdrant vector store
        """
        # set the storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex(
            nodes=self.nodes,
            embed_model=self.embed_model,
            storage_context=storage_context,
            show_progress=self.debug
        )

    @staticmethod
    def __set_llm(llm):
        """
        Set the LLM
        :return:
        """
        llm_settings = {
            'request_timeout': 60.0,
            'temperature': 1,
            'context_window': 4096
        }

        if llm == "mistral":
            llm_settings['context_window'] = 8192
        elif llm == "llama2":
            llm_settings['request_timeout'] = 30.0

        return Ollama(
            model=llm,
            **llm_settings
        )
