import logging

from llama_index.core.chat_engine.types import ChatMode
from utils.evaluation import (
    load_index,
    load_llm,
    check_if_evaluation_record_exists,
    write_evaluation_to_db,
    load_evaluation_dataset
)

# initialize logging for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_test(
        llm_model: str = "llama3_instruct",
        chat_mode: str = "CONDENSE_PLUS_CONTEXT",
        collection: str = "wiki_movie_plots_1024_100_mxbai",
        top_k: int = 2
):
    load_qna_dataset = load_evaluation_dataset()

    # Chatmodes
    chat_modes = {
        "CONDENSE_PLUS_CONTEXT": ChatMode.CONDENSE_PLUS_CONTEXT,
        "CONTEXT": ChatMode.CONTEXT
    }

    # init vector store index and query engine
    index = load_index(db="qdrant", collection_name=collection, debug=False)
    query_engine = index.as_chat_engine(
        chat_mode=chat_modes[chat_mode],
        llm=load_llm("oll_" + llm_model),
        similarity_top_k=top_k,
        verbose=True
    )

    # logging info
    logging.info(f"TEST INFO |:> llm_model: {llm_model} | collection_name: {collection}")

    index = 0
    for row in load_qna_dataset:
        context = row['context']
        if "complex" in row and row['complex']:
            continue
        for i, question in enumerate(row['questions']):
            # first check if row id is already in database
            exists = check_if_evaluation_record_exists(
                llm_model="oll_" + llm_model,
                collection_name=collection,
                question_id=index,
                chat_mode=chat_mode
            )

            # logging info
            if exists:
                logging.info(f"TEST INFO |:> Record exists: {index} | {exists} | {question}")
            elif not exists:
                logging.info(f"TEST INFO |:> Create record for: {index} | {question} | {row['context'][:30]}")

                # query the llm
                request = query_engine.chat(question)
                response = request.response

                # prepare data
                data = {
                    'llm_model': "oll_" + llm_model,
                    'collection_name': collection,
                    'question_id': index,
                    'question': question,
                    'context': context,
                    'reference_answer': row['reference_answers'][i],
                    'response': response,
                    'chat_mode': chat_mode
                }

                write_evaluation_to_db(data)

            # increase index
            index = index + 1


if __name__ == '__main__':
    collections = [
        "wiki_movie_plots_512_50_mxbai",
        "wiki_movie_plots_1024_100_mxbai",
        "wiki_movie_plots_2048_200_mxbai"
    ]
    llms = [
        "llama3_instruct",
        "gemma_instruct",
        "mistral_instruct"
    ]
    for collection_name in range(len(collections)):
        for llm in range(len(llms)):
            run_test(
                llm_model=llms[llm],
                chat_mode="CONTEXT",
                collection=collections[collection_name]
            )
