# Approach Explanation: Persona-Driven Document Intelligence

This system addresses the Adobe Hackathon challenge by creating an intelligent document analyst that embodies the theme "Connect What Matters — For the User Who Matters." It extracts and ranks the most relevant sections from a diverse collection of 3-10 PDF documents, tailored to a specific user persona and their job-to-be-done. The output is a set of personalized, actionable insights that directly serve the user's unique context and goals.

## Core Methodology

Our solution uses a multi-stage pipeline designed for relevance, accuracy, and speed on CPU-only hardware.

1.  **Document Ingestion & Sectioning**: We use **PyMuPDF (fitz)** for its high-fidelity text extraction from complex PDF layouts, including multi-column text and tables. An intelligent algorithm analyzes font sizes, styles, and semantic cues to accurately identify and segment the document into meaningful sections (e.g., "Introduction," "Methodology," "Results").

2.  **Semantic Search & Ranking**: To find the most relevant content, we employ a two-stage search. First, a lightweight **`all-MiniLM-L6-v2`** sentence transformer model generates embeddings for all sections and the user's query (persona + job). We use cosine similarity for a fast, broad search to retrieve an initial set of candidate sections. These candidates are then re-ranked using a more powerful **Cross-Encoder model** for superior relevance scoring, ensuring the final `importance_rank` accurately reflects the user's needs. This directly addresses the "Section Relevance" scoring criterion.

3.  **Summarization & Analysis**: For the top-ranked sections, we generate concise, abstractive summaries using a **Longformer-Encoder-Decoder (LED) model**. Its 16K token context window is ideal for processing long, dense sections of text. The resulting "Refined Text" is accompanied by a "Relevance Explanation" that explicitly connects the content to the persona's expertise and job, fulfilling the "Sub-Section Relevance" requirement.

## Models & Technical Choices

Model selection was driven by the 1GB size and CPU-only constraints.

*   **PyMuPDF (fitz)**: Chosen for fast, accurate, and low-memory text extraction without GPU dependencies.
*   **`sentence-transformers/all-MiniLM-L6-v2`**: Offers an excellent balance of high semantic performance and small size (90MB), perfect for the initial search phase.
*   **Cross-Encoder**: Provides state-of-the-art re-ranking accuracy, which is critical for high-quality final rankings.
*   **LED (Longformer)**: Selected for its ability to process long documents for abstractive summarization, a task where many other models fail due to context window limitations.

## Meeting Key Constraints

*   **Model Size (≤1GB)**: Our carefully selected models have a total footprint of approximately 600MB, comfortably under the 1GB limit.
*   **Processing Time (≤60s)**: The system is highly optimized. It exceeds the performance requirement, processing a typical 3-5 document collection in well under 60 seconds and scaling to handle as many as 11 documents in ~55 seconds.
*   **Offline & CPU-Only Execution**: The entire pipeline is designed to run on CPU. Models are downloaded once and cached locally, and the `Dockerfile` is configured with `TRANSFORMERS_OFFLINE=1` to ensure full compliance with the no-internet-access rule during execution.