# Text to SQL with Local LLM - Apartment Rental Database

## Project Overview
This repository implements a fully local Text-to-SQL system using Retrieval-Augmented Generation (RAG). Users can ask questions in natural language about apartment rentals, and the system automatically generates and executes the corresponding SQL query on a PostgreSQL database. The entire pipeline runs offline after initial setup, with no external API calls.

## Key Components and Exact Versions

| Component                  | Version          | Purpose |
|----------------------------|------------------|---------|
| Python                     | 3.13.5          | Main programming language |
| PostgreSQL                 | 15.8 (docker image: postgres:latest) | Relational database |
| Database name              | apartment_rentals | Stores all apartment rental data |
| Ollama                     | 0.3.13          | Local LLM server |
| LLM Model                  | llama3.1:8b     | Meta Llama 3.1 8B Instruct |
| Vanna.AI                   | 0.7.9           | Core RAG + Text-to-SQL framework |
| ChromaDB                   | 0.6.3           | Vector database (persisted in ./chroma_db) |
| Sentence-Transformers      | 5.1.1           | Embedding model: all-MiniLM-L6-v2 |
| Ollama Python client       | 0.6.0           | Communication with Ollama server (port 11434) |
| pandas                     | 2.2.3           | Data processing and evaluation |
| psycopg2-binary            | 2.9.11          | PostgreSQL connector |
| Gradio                     | 0.7.9           | Web interface |
| Plotly                     | 5.24.1          | Evaluation result visualization |

## Prerequisites
- Docker and Docker Compose: <https://docs.docker.com/engine/install/ubuntu/>

- Ollama: <https://ollama.com/download/linux>

- Python3.9+

## Installation Commands

```bash
# 1. Verify Python version
python3 --version

# 2. Deploy local PostgresDB with Docker Compose
./deploy_db.sh

# 3. Start Ollama
ollama serve
ollama pull llama3.1:8b

# 4. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate

# 5. Install python3 depdencies
pip3 install -r requirements.txt
```
## Execute text_to_SQL.py
```bash
python3 text_to_SQL.py
```
App will be deployed on <http://localhost:7860> and a gradio public URL show in the terminal.

In case you need to run only local you can fix the demo.lauch like below
```python
demo.launch(server_port=7860, inbrowser=True, share=False)
```
