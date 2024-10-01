# Thesis LLM: LLM + RAG
## Title

**Umsetzung innovativer NLP-Anwendungen auf Basis von Sprachmodellen und Retrieval Augmented Generation**

---
For the Bachlor of Science in Business Informatics at the University of Hochschule der Medien (HdM) in Stuttgart

## Setup of the Application
### System 
The whole Project was programmed on a MacBook Pro with macOS Sonoma 14.5 (23F79)
- Model: MacBook Pro 14-inch, 2021
- Chip: Apple M1 Pro
- Memory: 16 GB
- Storage: 1 TB

### Packages and Versions
- Python 3.10.14
- Docker version 26.1.1, build 4cf5afa
- Docker Compose version v2.27.0-desktop.2
- ollama version is 0.1.39
- python=3.8.11
- conda-forge::streamlit==1.35.0
- pytorch::pytorch==2.3.1
- pytorch::torchvision==0.18.1
- pytorch::torchaudio== 2.3.1
- pytorch::pytorch-cuda==12.1
- jupyter==1.0.0
- matplotlib==3.7.5
- pandas==2.0.3
- numpy==1.24.4
- seaborn==0.13.2
- python-dotenv==1.0.1
- chromadb==0.5.0
- llama-index==0.10.44
- llama-index-core==0.10.44
- llama-index-llms-ollama==0.1.5
- llama-index-vector-stores-chroma==0.1.9
- llama-index-embeddings-huggingface==0.2.1
- llama-index-vector-stores-qdrant==0.1.3

### Prerequisites for the Installation
- Python 3.10 - https://www.python.org/downloads/release/python-3100/
- Conda or Pip - https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
- Docker - https://docs.docker.com/compose/install/
- Docker-Compose - https://docs.docker.com/compose/install/
- qdrant - llama-index-vector-stores-qdrant
- chromadb - llama-index-vector-stores-chromadb
- ollama - https://ollama.com/

### Installation with Conda

Creating an _environment.yml_ to install packages with conda
```sh
conda env create -f environment.yml
```

#### Create an Environment with Conda
```sh
conda create --name thesis_llm -c conda-forge python=3.10
```
#### Activate all the installed packages for the environment used for this application
```sh
conda activate thesis_llm
```
#### Update or install Packages that are listed in _environment.yml_
```sh
conda update --all
```
```sh
conda env update --file environment.yml
```
To deactivate this environment of this application just run the following command
```sh
conda deactivate
```

### Installation with PIP
if you want to install the repo with _pip_ and use a _requirements.txt_ just run "requirements.py".
It's creating a _requirements.txt_ from the _environment.yml_
```sh
python requirements.py
```
### Qdrant

Start docker container
```sh
docker compose up --remove-orphans -d
```
```sh
docker-compose restart qdrant
```
WebUI: http://localhost:6333/dashboard

```sh
docker-compose exec qdrant bash
```

```sh
docker compose down
```
```sh
docker volume rm $(docker volume ls -qf dangling=true)
```

### Indexing: creating Vector Index
```sh
PYTHONPATH=$(pwd) /opt/miniconda3/envs/thesis_llm/bin/python ingest.py
```
## Chatbot
### Serve Ollama
```sh
ollama serve
```
how to kill process
```sh
kill -9 $(lsof -t -i:11434)
```
### Download Models used in the Project
Model: [gemma:instruct](https://ollama.com/library/gemma) model for follow instructions
```sh
ollama run gemma:instruct
```
Model: [llama3:instruct](https://ollama.com/library/llama3:instruct) model for follow instructions
```sh
ollama run llama3:instruct
```
Model: [mistral:instruct](https://ollama.com/library/mistral:instruct) model for follow instructions
```sh
ollama run mistral:instruct
```
### Start the Chatbot
MacOS
```sh
/opt/miniconda3/envs/thesis_llm/bin/python -m streamlit run chatbot.py
```
Windows
```sh
streamlit run chatbot.py
```

## Credits
https://stackoverflow.com/questions/62630186/anaconda-always-want-to-replace-my-gpu-pytorch-version-to-cpu-pytorch-version-wh/62633536#62633536

https://stackoverflow.com/questions/69694093/gpu-is-not-available-for-pytorch