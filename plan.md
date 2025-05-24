# The Plan


### 24.05.2025

- Done

Got USC 26 data, rough langchain RAG setup, started parsing USC 26, came up with ideas to make project broad.

- To Do

1. Get German tax law (federal at first)

2. Parse USC 26 and German tax law (deal with silly structures like tables)

3. Get basic Langchain RAG pipeline working for each (stock parameters and LLMs)

4. Figure out how to evaluate the models (VDB retrieval, so precision and recall -- qualitative analysis of generation against baseline LLM)

5. Debug any weirdness with the generation

6. Look at changing variables (embedding model, base LLM, data?)

6. Look at fine tuning the embedding model

7. Prepare presentation

- Blockers



## Presentation outline - subject to change

I. Motivation
    Tax law is complex, full of cross-references.

    ChatGPT hallucinations = risky.

    RAG helps ground responses in real law & cite exact sections.

II. Background Crashcourse
    Embeddings
    
    Transformers

    BERT

III. Pipeline Overview
    🔄 Data Sources: Federal code (XML), IRS pubs (PDF)

    🔍 Vectorization with FAISS + OpenAI embeddings

    🧠 RAG chain with LangChain

    💬 QA over structured knowledge

IV. Technical Highlights
    Chunking strategy (section-wise vs. sliding window)

    Embedding strategy and potential fine-tuning

    Example query evaluation (e.g., rental income, LLC taxation)

V. Results
    ~80–90% accurate answers on legal-style tax questions

    Areas for improvement: citation accuracy, nuanced exceptions

VI. Extensions
    Add citations with source_documents

    Use map_reduce chain for long documents

    Fine-tune embeddings or LLM on IRS/IRC datasets

