import asyncio
import logging
import streamlit as st

# https://github.com/run-llama/llama_index/issues/7244
# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

# load components form utils
from utils.chatbot import chatbot
from utils.components import page_config, add_sidebar, chatbot_intro, styling, load_model
from utils.config import DEBUG

# initialize logging for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# @credits
# https://www.youtube.com/watch?v=nF-PQj0k5-o
# https://github.com/iamaziz/ollachat/blob/main/ollachat/chatbot.py
# https://discuss.streamlit.io/t/build-your-own-notion-chatbot/51497
# https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
# https://medium.com/@maximejabarian/building-a-local-llms-app-with-streamlit-and-ollama-llama3-phi3-511d519c95fe

async def main():
    # PAGE CONFIG
    page_config()

    # LOAD STYLES
    styling()

    # SIDEBAR
    sidebar_settings = add_sidebar()
    llm = load_model(sidebar_settings["llm"])

    # LOAD INTRO
    chatbot_intro()

    # DEBUG
    if DEBUG:
        with st.expander("DEBUG Details"):
            st.session_state

    # CHATBOT
    chatbot(sidebar_settings)


if __name__ == '__main__':
    # run chatbot
    asyncio.run(main())
