import json
from pathlib import Path

from llama_index.core import Document

from utils.config import OUTPUT_FILE

# Sources:
# https://docs.llamaindex.ai/en/stable/understanding/loading/loading/
# https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/#documents-nodes
# https://docs.llamaindex.ai/en/stable/api_reference/readers/json/?h=jsonreader#llama_index.readers.json.JSONReader
# https://saturncloud.io/blog/how-to-check-if-a-variable-is-either-a-python-list-numpy-array-or-pandas-series/
# https://docs.python.org/3/tutorial/errors.html#raising-exceptions
# https://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod-in-python
# https://www.geeksforgeeks.org/private-methods-in-python/

"""
The Documents class is responsible for loading the data from the JSON file
- Load the data from the JSON file
- Create document objects
"""


class Loading:

    def __init__(self, input_file=OUTPUT_FILE, formatted=True):
        self.input_file = input_file
        self.json_data = self.read_input_json_file()
        self.documents = []
        self.text_formatted = formatted
        self.text_template = "Metadata:\n{metadata_str}\n-----------\nContent:\n{content}"
        self.include_from_metadata = ['release_year', 'origin_ethnicity', 'plot_length', 'title']
        self.metadata_template = "{key}: {value}"
        self.metadata = {
            'extension': '.json',
            'file_name': Path(self.input_file).name
        }

    def read_input_json_file(self):
        """
        Load the data from the JSON file
        :return:
        """
        # init the input file as a Path object
        file_path = Path(self.input_file)

        # raise an error if the file does not exist
        if not file_path.exists():
            raise FileNotFoundError(f"No file found at {file_path}")

        # read the data from the input file
        with file_path.open('r', encoding="utf-8") as file:
            data = json.load(file)
            if not data:
                raise ValueError(f"No data found in {file_path}")

            if not isinstance(data, list):
                raise ValueError(f"Data in {file_path} must be a list")
        return data

    def get_json_data(self):
        """
        Get the JSON data
        :return:
        """
        return self.json_data

    def load_data(self):
        """
        Load the data from the JSON file
        and create a list of documents
        :return:
        """
        for row in self.get_json_data():
            self.documents.append(self.__create_document(row))

    def get_documents(self):
        """
        Get the list of documents
        :return:
        """
        self.load_data()
        return self.documents

    def get_training_test_data(self):
        """
        Get the training and test data
        :return:
        """
        # randomly shuffle the data
        import random
        random.shuffle(self.get_json_data())

        # split the data into training and test data
        training_data = []
        test_data = []
        # split 80% training and 20% test

        split = int(0.8 * len(self.get_json_data()))
        print(split)
        for i, row in enumerate(self.get_json_data()):
            create_document = self.__create_document(row)
            if i < split:
                training_data.append(create_document)
            else:
                test_data.append(create_document)
        return training_data, test_data

    def __create_document(self, document):
        """
        Create a document object
        :param document:
        :return:
        """
        if self.text_formatted:
            text = self.__create_document_text(document)
        else:
            text = json.dumps(document)
        return Document(
            text=text,
            metadata=self.__create_metadata(document),
            excluded_llm_metadata_keys=["file_name"],
            text_template=self.text_template
        )

    def __create_metadata(self, row):
        """
        Create the metadata for the document
        :param row:
        :return:
        """
        metadata = {}
        for key, value in row.items():
            if key in self.include_from_metadata:
                metadata[key] = value
        return metadata

    @staticmethod
    def __create_document_text(row):
        """
        Create the text for the document
        :param row:
        :return:
        """
        content = ""
        for key, value in row.items():
            content += f"{key}: {value}\n"
        return content
