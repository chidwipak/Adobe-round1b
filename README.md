# Persona-Driven Document Intelligence

## Project Description

A sophisticated, offline-capable document analysis system designed for the Adobe Round 1B Hackathon that intelligently extracts and ranks the most relevant sections from 3-10 diverse PDF documents based on a specific persona and job-to-be-done. The system embodies the theme *"Connect What Matters — For the User Who Matters"* by delivering personalized, actionable insights that directly serve the user's specific needs and professional context.

## Features \& Outcomes

- **🎯 Persona-Driven Analysis:** Extracts sections most relevant to specific user personas and job requirements
- **📊 Intelligent Ranking:** Uses semantic similarity to rank document sections by relevance to persona and job
- **🔍 Subsection Refinement:** Provides meaningful summaries with detailed relevance explanations
- **⚡ CPU-Optimized:** Runs entirely on CPU with no GPU requirements
- **📦 Compact Models:** Total model size under 1GB (actual ~600MB)
- **🚀 Fast Processing:** Processes 11 documents in ~55 seconds
- **🔒 Offline Capable:** Works completely offline after initial model download
- **🎨 Diverse Document Support:** Handles academic papers, technical docs, resumes, and more


## Input \& Output

### Input Format

- **PDF Documents:** Place 3-10 diverse PDF files in the `input/` directory
- **Persona Description:** JSON file with role, expertise, and interests (optional; auto-generated otherwise)
- **Job-to-be-Done:** Specific task or objective tied to the persona


### Output Format

The system generates a comprehensive JSON file at `output/persona_analysis.json` including:

- **Metadata:** Processing timestamps, document statistics, persona details
- **Extracted Sections:** Top 10 most relevant sections with importance rankings
- **Subsection Analysis:** Refined text summaries and detailed relevance explanations


## Setup \& Installation

### System Requirements

- Python 3.8 or higher
- Minimum 4GB RAM (8GB recommended)
- At least 2GB free disk space
- Internet connection **only for initial model download**


### Installation Options

#### Option 1: Docker (Recommended)

```bash
# Build the Docker image
docker build -t document-intelligence .

# Run the container with input/output directories mounted, no internet access

docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none connecting-dots:latest
```


#### Option 2: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd adobe-1b

# Install dependencies
pip install -r requirements.txt

# Set Hugging Face token (replace with your actual token)
export HF_TOKEN=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
```


## Usage Instructions

### Basic Usage

```bash
python3 src/cli.py --input-dir input --output-dir output --persona-file output/persona.json
```


### Advanced Options

```bash
# Clean up downloaded models to free space
python3 src/cli.py --input-dir input --output-dir output --cleanup-models

# Enable verbose logging for debugging
python3 src/cli.py --input-dir input --output-dir output --verbose
```


### Input Preparation

1. Place all PDF documents in the `input/` directory
2. Ensure the `output/` directory exists
3. Provide a `persona.json` file in `output/` directory, or allow the system to auto-generate one

### Output Location

- Main Output JSON: `output/persona_analysis.json`
- Persona JSON: `output/persona.json` (auto-generated if not provided)
- Logs: Console output


## Performance \& Constraints

| Metric | Actual | Constraint |
| :-- | :-- | :-- |
| Processing Time | ~55 seconds | ≤ 60 seconds |
| Model Size | ~600 MB | ≤ 1 GB |
| Memory Usage | ~2 GB peak | (No specific limit) |
| Execution Mode | CPU only | CPU only |
| Internet Required | Only initially | No internet post-download |

## Model Management

### Automatic Model Download

Models are downloaded securely using your HuggingFace token:

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)
- **Summarization model:** LED (Longformer-Encoder-Decoder)
- **Cross-Encoder:** For enhanced relevance scoring and ranking


### Caching \& Cleanup

- Models are cached locally under `/opt/models` (configurable)
- Subsequent runs operate offline using cached models
- Cleanup command available to delete cached models and free disk space

```bash
python3 src/cli.py --cleanup-models
du -sh /opt/models  # check disk usage
```


## Architecture \& Components

### Core Pipeline Steps

1. **Document Processing:** Extract text and sections from PDFs using PyMuPDF
2. **Semantic Encoding:** Generate embeddings with `all-MiniLM-L6-v2` for persona+job and document sections
3. **Section Ranking:** Compute cosine similarity and apply diversity-aware selection
4. **Subsection Analysis:** Generate summaries using hybrid LED summarization and TextRank
5. **Output Generation:** Construct output JSON complying with Adobe Round 1B schema

### Key Files

- `src/pipeline_r1b.py` — Main processing pipeline
- `src/model_utils.py` — Model loading and inference utilities
- `src/pdf_extract.py` — PDF text extraction and section splitting
- `src/cli.py` — Command-line interface entrypoint
- `Dockerfile` — Container setup
- `requirements.txt` — Python dependencies list


## Troubleshooting

### Model Download Issues

```bash
# Verify HF token set
echo $HF_TOKEN

# Test API access
curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2
```


### Memory or Performance Tips

```bash
# Lower batch size for large docs
export BATCH_SIZE=1

# Monitor CPU/memory usage
htop
```


### Logging \& Debugging

Use `--verbose` flag for detailed logs:

- Model loading and inference steps
- Text extraction progress
- Similarity scoring
- Summary generation verification


## Evaluation \& Quality Assurance

- Ensures **clear section titles** and meaningful **refined summaries**
- Provides **specific relevance explanations** directly tied to persona/job
- Validates **embedding quality** to avoid empty or zero vectors
- Guarantees **compliance to Adobe JSON schema** and timing constraints


## Contributing \& Extending

### Adding Custom Personas

Modify `persona.json` format as needed; ensure role and job fields are descriptive.

### Integrating New Models

- Update model definitions in `model_utils.py`
- Adjust summarization or ranking logic in `pipeline_r1b.py`


## License \& Acknowledgements

Developed for Adobe Round 1B Hackathon leveraging:

- [PyMuPDF](https://pymupdf.readthedocs.io) for PDF processing
- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for summarization
- Python ecosystem libraries for data handling and processing


## Support \& Documentation

- See `approach_explanation.md` for detailed methodology and design decisions
- Check example outputs in `output/` for validation
- For questions, contact the author or raise issues in the repository


