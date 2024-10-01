from llama_index.core.evaluation.answer_relevancy import AnswerRelevancyEvaluator

from utils.evaluation import load_llm, load_index

llm_model = load_llm("oll_llama3_instruct")
evaluator = AnswerRelevancyEvaluator(llm=llm_model)

index = load_index(collection_name="wiki_movie_plots_1024_100_mxbai", debug=True)
query_engine = index.as_query_engine(llm=llm_model)
#query = "Who is the director of the movie \"The Karate Kid\" released in 1984?"
#query = "What is the plot of \"The Karate Kid\" released in 1984?"
query = "What is the genre of the movie \"The Karate Kid\" released in 1984?"
response = query_engine.query(query)
context = """
    release_year: 1984
    title: The Karate Kid
    origin_ethnicity: American
    director: John G. Avildsen
    cast: Ralph Macchio, Pat Morita, Elisabeth Shue, William Zabka, Martin Kove
    genre: drama
    wiki_page: https://en.wikipedia.org/wiki/The_Karate_Kid_(1984_film)
    plot: Daniel LaRusso and his mother move from Parsippany-Troy Hills, New Jersey to Reseda, Los Angeles, California. The maintenance man in their new apartment complex is an eccentric, kind and generous Okinawan immigrant named Mr. Miyagi.
    At a beach party, Daniel meets Ali Mills, a high school cheerleader from Encino, Los Angeles. Johnny Lawrence, Ali's ex-boyfriend, is the top student of a karate dojo called "Cobra Kai." When Johnny deliberately breaks Ali's radio, Daniel attempts to stop him, but is easily overpowered and humiliated by Johnny. Continuously bullied by the Cobra Kai after this, Daniel finds solace with Miyagi. At a Halloween party, Daniel douses Johnny with water; chased and eventually cornered by Johnny and his accomplices, Daniel is savagely beaten until Miyagi intervenes, easily disabling the attackers.
    Daniel asks Miyagi to teach him to fight. Miyagi refuses, instead agreeing to accompany Daniel to the Cobra Kai dojo to resolve the conflict. They meet with the sensei, John Kreese, an ex-Special Forces Vietnam veteran who teaches his students to be aggressive and merciless against their opponents. He dismisses the peace offering made by Miyagi, so Miyagi proposes that Daniel will enter the Under-18 All-Valley Karate Tournament, where he will compete against the Cobra Kai students, and requests that the bullying cease while Daniel trains. Kreese agrees to the terms, and warns that if Daniel does not appear at the tournament, the harassment will resume on both Daniel and Miyagi.
    Daniel's 'training' starts under the guise of having him complete various lengthy, menial chores that appear to have nothing to do with karate, but which are actually teaching muscle memory. Through Miyagi's teaching, Daniel learns the necessity of personal balance, reflected in the principle that martial arts training is not so much about disciplining the body as it is the spirit.
    At the tournament, Daniel unexpectedly reaches the semi-finals. After Daniel defeats a particularly skilled opponent, Kreese, worried that Daniel might make it to the finals, instructs Bobby Brown—one of his more compassionate students and the least vicious of Daniel's tormentors—to disable Daniel with an illegal attack to the knee. Bobby reluctantly does so and is disqualified. Daniel, refusing to concede, convinces Miyagi to use a pain suppression technique so he can continue the tournament. Daniel, barely able to stand, uses a Crane kick, which allows him to deliver a blow to Johnny's head using only one leg and wins the tournament. Johnny, having gained respect for his nemesis, gives Daniel his trophy and Daniel is carried off by the enthusiastic crowd.
    plot_length: 2672
    """

result = evaluator.evaluate(
    query=query,
    contexts=[context],
    response=response.response
)
score = result.score
breakpoint = "debug"

## Note:
## Use Evaluator: yes
## Result:
# context: none
# feedback: Let's evaluate the response:
#
# 1. Does the provided response match the subject matter of the user's query?
#
# Yes, the response mentions the director of the movie "The Karate Kid", which is directly related to the query.
#
# 2. Does the provided response attempt to address the focus or perspective on the subject matter taken on by the user's query?
#
# Yes, the response provides a specific answer to the question about the director of the movie, which addresses the focus of the query.
#
# Total score: 2/2
#
# [RESULT] Relevant followed by 2

# passing: none
# query: Who is the director of the movie "The Karate Kid" released in 1984?
# response: John G. Avildsen
# score: none