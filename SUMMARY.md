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

## Next Steps

1. **Scale up post data** to 30+ posts (5 for now) and validate retrieval quality with diverse queries
2. **Handle images in posts** using vision language models 
3. **API integration** with existing student post retrieval 

## Final Codebase

### Essential Files (6 files)

1. **`rag_system.py`** - Main RAG system
2. **`pdf_processor.py`** - PDF extraction with markdown conversion
3. **`cost_estimator.py`** - Cost tracking
4. **`config_loader.py`** - Secure config management
5. **`demo.py`** - Complete demonstration

### Configuration Files
- `config.json` - API keys (gitignored)
- `config.json.template` - Setup template
- `.gitignore` - Security

## Demo Results

Uses posts from `posts/dashboard.json` (5 posts, 1 PDF)

**Query:** "When can we get assignment 2 solutions?"
→ Returns POST "a2 solutions #224" (with a2_sol.pdf attachment)

## How to Run

```bash
# Setup
cp config.json.template config.json
# Add OpenAI API key to config.json

# Run demo
python3 demo.py
```

## Integration with Ed Platform

```python
from rag_system import PostRAGSystem

# Initialize
rag = PostRAGSystem()

# Add posts (from interface folder or posts/dashboard.json)
for post in posts:
    rag.add_post(post, base_dir='posts')

# Query
results = rag.retrieve("student question", n_results=5)
```

