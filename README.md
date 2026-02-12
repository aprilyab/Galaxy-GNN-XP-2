# 🌌 Galaxy Workflow Tool Prediction – GNN & LSTM Recommender

Galaxy-GNN-XP-2 is a professional tool prediction pipeline designed to enhance the Galaxy workflow experience. By leveraging **Neo4j graph data** and **Deep Learning (LSTM)**, this project predicts the most logical "next tool" in a bioinformatics sequence, streamlining workflow construction for researchers.

##  Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Security & Configuration](#security--configuration)
- [Data Pipeline](#data-pipeline)
  - [Step 1: Neo4j Extraction](#step-1-neo4j-extraction)
  - [Step 2: Sequence Preparation](#step-2-sequence-preparation)
  - [Step 3: Model Training and Evaluation](#step-3-model-training-and-evaluation)
- [Technical Stack](#technical-stack)
- [Testing & Verification](#testing--verification)

---

##  Features
- **Graph-Powered Extraction**: Paginated extraction logic to pull high-quality workflow sequences from Neo4j.
- **Deterministic Sequence Building**: Advanced topological sorting and DFS algorithms to handle branching and divergent workflow paths.
- **Semantic Normalization**: Automated cleaning and standardizing of Galaxy tool identifiers for robust vocabulary management.
- **Bi-directional LSTM Architecture**: High-capacity sequence models trained on processed Galaxy tool chains.
- **Interactive Benchmarking**: A dedicated evaluation module to calculate Precision, Recall, Hit@5, and MRR.
- **Full Unit Test Suite**: 100% coverage of core business logic to ensure data integrity.

##  Project Structure
```text
src/
├── workflow_extraction.py   # Entry: Neo4j -> JSON sequence extraction
├── sequence_preparation.py  # Entry: Tool cleaning, vocab building, tensor generation
├── neo4j_connection.py      # Infrastructure: Connectivity & Cypher queries
├── sequence_generation.py   # Business Logic: Sequence construction (DFS/Topological)
├── schema_models.py         # Data Systems: Pydantic schemas & validation
└── utils.py                 # Core Helpers: Logging, I/O, Vocab, and Datasets
```

## Quick Start

### 1️ Prerequisites
- **Python** 3.10+
- **Neo4j Instance** (Local or Remote)


### Installation
```bash
# Clone the repository
git clone https://github.com/henok/Galaxy-GNN-XP-2.git
cd Galaxy-GNN-XP-2

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Security & Configuration
Set your Neo4j credentials as environment variables or pass them directly to the extraction script:
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_secure_password"
```
---

##  Data Pipeline

### Step 1: Neo4j Extraction
Convert raw graph workflows into structured JSON sequences.
```bash
python3 -m src.workflow_extraction --uri $NEO4J_URI --user $NEO4J_USER --password $NEO4J_PASSWORD
```

### Step 2: Sequence Preparation
Perform tool normalization, build the indexing vocabulary, and generate PyTorch tensors.
```bash
python3 -m src.sequence_preparation  # TO Flatten graph data to sequences
```

### Step 3: Model Training and Evaluation
Run the interactive benchmarking module to visualize performance metrics.
```bash
# Launch Jupyter and run benchmarking/benchmark_lstm.ipynb
jupyter notebook benchmarking/benchmark_lstm.ipynb
```
---

##  Technical Stack
- **Deep Learning**: PyTorch LSTM
- **Database**: Neo4j (Cypher)
- **Data Validation**: Pydantic
- **Evaluation**: Scikit-Learn, Matplotlib, Numpy
- **Environment**: Python 3.10+, Jupyter

##  Testing & Verification
We maintain a strict quality standard. Run the verified unit test suite with:
```bash
python3 -m unittest discover tests
```
