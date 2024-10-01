import logging

from utils.evaluation import create_tables

# initialize logging for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    create_tables()
