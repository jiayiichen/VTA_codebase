# VTA RAG System 
# 10.19. Week Deliverable Summary

## What Was Built

A RAG system for VTA that:
- Uses VTA interface to retrieve Ed discussion posts and PDF attachments
- Embeds posts and PDFs separately using OpenAI text-embedding-3-small
- Uses cosine similarity for semantic matching (scores 0-1, higher is better, threshold = 0.5)
- Returns: (1) most relevant post (always a post, includes PDF info when matched via PDF), (2) similar student posts (with similarity ≥ 0.5), (3) course materials
- Shows best match even when below threshold with low-confidence warning
- Prevents duplicate processing by checking if posts already exist in database

## Next Steps (update after 10.20. meeting)

1. Adjust similarity threshold to 0.7
2. Handle images in posts using vision language models 

## Final Codebase

### Essential Files (6 files)

1. **`rag_system.py`** - Main RAG system
2. **`pdf_processor.py`** - PDF extraction with markdown conversion
3. **`cost_estimator.py`** - Cost tracking
4. **`config_loader.py`** - Secure config management
5. **`demo.py`** - Complete demonstration

## How to Run

```bash
# Setup
cp config.json.template config.json
# Add OpenAI API key to config.json

# Run demo
python3 demo.py
```

# 10.26. Week Deliverable Summary

## What Was Built

Added ColPali visual retrieval system that:
- Uses ColQwen2-v1.0 vision model for image-based PDF and image processing
- Hybrid approach: text posts use OpenAI embeddings (same as before), PDFs/images use ColQwen2 vision embeddings
- Stores embeddings in pickle files (texts.pkl for text, images.pkl for visual content)
- Auto-detects file types (PDF vs image) and processes accordingly
- Optimized for GPU (CUDA) with CPU fallback for Apple Silicon

## System Comparison

| Feature | Original (rag_system.py) | ColPali (colpali_rag_system.py) |
|---------|-------------------------|----------------------------------|
| Post text | OpenAI embeddings | OpenAI embeddings (same) |
| PDFs | Text extraction → OpenAI | Images → ColQwen2 vision |
| Images | Filename only | Full visual processing |
| Speed | Fast (seconds) | Slow CPU / Fast GPU |
| Cost | ~$0.02/year | $0 (local) |
| Storage | ChromaDB | Pickle files |
