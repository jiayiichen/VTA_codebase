ColPali for Google Colab - Upload Guide
========================================

CONTENTS OF THIS FOLDER:
-------------------------
1. colpali_processor.py       - ColQwen2 vision model processor
2. colpali_rag_system.py      - Hybrid RAG system (text + images)
3. config_loader.py           - Configuration loader
4. config.json                - Your OpenAI API key (KEEP PRIVATE!)
5. colpali_demo_colab.ipynb   - The notebook to run
6. posts/                     - Your posts and attachments data
7. colpali_rag_db/            - Pre-processed embeddings (optional)

STEPS TO USE IN COLAB:
----------------------
1. Go to https://colab.research.google.com
2. Upload colpali_demo_colab.ipynb (File → Upload notebook)
3. Set GPU: Runtime → Change runtime type → Hardware accelerator → GPU (T4)
4. Mount Google Drive and upload this entire folder to MyDrive/colpali_colab_upload/
5. Run the notebook cells in order
   - Cell 2 installs poppler-utils (required for PDF processing)
   - Cell 4 mounts Google Drive and changes to your folder

IMPORTANT NOTES:
----------------
✓ NO path changes needed - all files use relative paths
✓ Keep files in the same directory structure
✓ Make sure config.json has your OpenAI key
✓ GPU will make processing 50-100x faster than CPU
✓ First run downloads ~2-3GB ColQwen2 model (one-time)

OPTIONAL - Skip Processing:
---------------------------
If you upload the colpali_rag_db/ folder, you can:
- Skip processing and test queries immediately
- Saves time on first run
- Contains 5 posts + 4 images already processed

FILE SIZE:
----------
Total: ~4MB (without model, which downloads automatically)

HELP:
-----
If something doesn't work, check:
- GPU is enabled in Runtime settings
- config.json has valid OpenAI API key
- All files uploaded to same directory in Colab
