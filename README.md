# Sustainable Travel Advisor

This project aims to build a Retrieval-Augmented Generation (RAG) system for eco-conscious travelers. The system will eventually provide local, sustainable travel advice by retrieving relevant information from curated city sustainability guides, green accommodation directories, and responsible dining and transportation resources.

## Week 2 Deliverables
- Defined project domain and target users
- Documented a clear problem statement and expected user questions
- Established the required project folder structure
- Created a Week 2 domain plan for Week 3 implementation

## Week 3 Progress
- Added a modular document preparation pipeline for loading, cleaning, chunking, and saving source documents
- Implemented reusable source modules in the src folder:
  - loader.py for reading PDF files
  - cleaner.py for text normalization and paragraph cleaning
  - chunker.py for paragraph-based chunk creation with metadata
  - utils.py for logging, directory creation, and file export helpers
- Created the runnable pipeline entry point in run_pipeline.py
- Generated the required output files in data/chunks:
  - chunks.json
  - chunks.csv
- Added requirements.txt for the PDF dependency
- Included a notebook demonstration in notebooks/week3_pipeline_demo.ipynb

## How to Run Week 3 Pipeline
1. Place PDF files into data/raw_documents/
2. Install dependencies with: pip install -r requirements.txt
3. Run: python run_pipeline.py

## Notes
- This week focuses only on document loading, preprocessing, cleaning, chunking, metadata creation, and saving.
- Embeddings, vector stores, semantic search, and retrieval are intentionally not implemented in Week 3.
