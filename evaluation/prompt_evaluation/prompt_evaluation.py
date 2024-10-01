import os.path

from llama_index.core.chat_engine.types import ChatMode

from utils.config import ROOT_DIR
from utils.evaluation import load_llm, load_index
from utils.evaluation import get_records_by_question_ids


def get_random_question_ids():
    random_question_ids = f'{ROOT_DIR}/evaluation/evaluation_tests/random_question_ids.txt'
    # read the file and return the question ids as list of integers
    with open(random_question_ids, 'r') as f:
        question_ids = f.readlines()
        print()
    return [int(q_id.strip()) for q_id in question_ids[0].split(', ')]


def run_evaluation():
    # list with ids
    list_with_ids = get_random_question_ids()

    # get records by question ids
    random_question_data = get_records_by_question_ids(list_with_ids)

    # extract the questions from the records
    list_with_questions = [record[5] for record in random_question_data]

    # define collections
    collections = {
        "wiki_movie_plots_512_50_mxbai": "Wiki Movie Plots 512 50 mxbai",
        "wiki_movie_plots_1024_100_mxbai": "Wiki Movie Plots 1024 100 mxbai",
        "wiki_movie_plots_2048_200_mxbai": "Wiki Movie Plots 2048 200 mxbai"
    }

    # define llm models
    llms = [
        "llama3_instruct",
        "gemma_instruct",
        "mistral_instruct"
    ]

    # prompt templates
    prompt_templates = {
        0: (
            # Small and Simple prompt
            "You are a Movie Expert. Keep your responses simple. "  # Role + Text Style
            "Structure your output in a way that is easy to read and understand."  # Formatting
        ),
        1: (
            # Small and Simple prompt but more information about the movie
            "You are a Video Librarian you want to recommend movies to others. "  # Role
            "But you don't want to spoil. Be kind "  # Style
            "and don't give away too much. Keep your responses simple and to the point."  # Formatting
            "Talk about the movie's genre, plot, and main characters to arouse the interest of the user "  # Content
            # Style
        ),
        2: (
            # More interaction by asking the user if they want to know more about the movie
            "You are very knowledgeable about movies. "  # Role
            "You want to share your knowledge with others. "  # Style
            "Provide detailed information about the movie's genre, plot, and main characters. "  # Content
            "Use bullet points, lists, and other formatting to make your responses more readable."  # Formatting
            "Take all the Context and ask if the user wants to know more about the movie."  # Content + Style
            # more Interaction
        ),
        3: (
            # Put everything together many details very clear and detailed prompt
            "You are a movie expert. Your primary responsibility is to assist users in discovering and learning about"
            " films. You will have context with information about the movie. "  # Role + Content
            "Use this information to provide detailed, accurate, and engaging information about the movie. "  # Content
            "Work with bullet points, lists, and other formatting to make your responses more readable."  # Formatting
            "Talk about the movie's genre, plot, and main characters to arouse the interest of the user."  # Content
            # Style
            "If there is nothing given in the Context then please under any circumstances do not provide "
            "any information. Arouse the interest of the user and ask if they want to know more about the movie."
            # Content
        ),
        4: (
            # Now Short but detailed
            "You are a recommender. The User want to Discover Movies or Films. Use provided Context. "
            "Structure your Output (bullet-points). Arouse the users interest. Don't Spoil. "
            "Ask if user want to now more."  # Role
        )
    }

    extra_questions = [
        "Recommend an action movie",
        "I want to watch a movie with fast cars and explosions",
        "I want to watch a movie with a Superhero",
    ]

    for question in list_with_questions + extra_questions:
        for prompt in prompt_templates:
            for collection in collections:
                for llm_model in llms:
                    file_name = f'{ROOT_DIR}/evaluation/prompt_evaluation/results/{llm_model}_{collection}.md'
                    llm = load_llm('oll_' + llm_model)
                    index = load_index(collection_name=collection, debug=True)
                    query_engine = index.as_chat_engine(
                        system_prompt=prompt,
                        chat_mode=ChatMode.CONTEXT,
                        llm=llm,
                        similarity_top_k=2
                    )

                    rspo = query_engine.chat(
                        message=question
                    )

                    if len(rspo.source_nodes) == 0:
                        nodes = 'No nodes found'
                    else:
                        nodes = ', '.join([f"{round(n.score, 2)}: {n.metadata['title']}" for n in rspo.source_nodes])

                    # check if file exists and append the prompt, query and response
                    with open(file_name, 'a') as f:
                        append = '\n' + \
                                 f'-' * 50 + '\n' + \
                                 f'Iteration: {prompt}\n' + \
                                 f'Collection: {collection}\n' + \
                                 f'Prompt: {prompt_templates[prompt]}\n' + \
                                 f'Query: {question}\n' + \
                                 f'Response: {rspo.response}\n' + \
                                 f'Nodes: {nodes}\n'
                        f.write(append)
                        f.close()


if __name__ == '__main__':
    run_evaluation()
