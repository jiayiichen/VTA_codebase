"""
ColPali Processor for VTA RAG System

Uses colqwen2-v1.0 vision model to process PDFs and images,

Optimized for GPU (CUDA) and CPU fallback.
"""

import os
import time
from typing import List, Dict, Optional
from PIL import Image
import torch
from pdf2image import convert_from_path

try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False


class ColPaliProcessor:
    """Process PDFs and images using ColQwen2 vision model for visual retrieval."""

    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", device: str = "auto"):
        """
        Initialize ColPali processor.

        Args:
            model_name: HuggingFace model name (default: vidore/colqwen2-v1.0)
            device: Device to use ('auto', 'mps', 'cuda', 'cpu')
        """
        if not COLPALI_AVAILABLE:
            raise ImportError("ColPali not installed. Run: pip install colpali-engine")

        self.model_name = model_name
        self.device = self._setup_device(device)

        # Load model and processor
        start_time = time.time()

        if self.device == "mps":
            dtype = torch.float16
        elif self.device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        try:
            if self.device == "cuda":
                # For CUDA, use device_map for efficient loading
                self.model = ColQwen2.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=self.device
                )
            else:
                # For MPS/CPU, load directly without device_map
                self.model = ColQwen2.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                )
                # Model is already on CPU after loading, no need to move

            self.processor = ColQwen2Processor.from_pretrained(model_name)

        except Exception as e:
            self.device = "cpu"
            self.model = ColQwen2.from_pretrained(model_name, torch_dtype=torch.float32)
            self.processor = ColQwen2Processor.from_pretrained(model_name)

    def _setup_device(self, device: str) -> str:
        """Determine the best available device."""
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to PIL images.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion (default: 200)

        Returns:
            List of PIL Images (one per page)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            return images
        except Exception:
            return []

    def generate_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Generate ColPali embeddings for a list of images.

        Args:
            images: List of PIL Images

        Returns:
            Tensor of shape (num_pages, num_patches, embed_dim)
        """
        if not images:
            return torch.tensor([])

        # Process images through ColPali processor
        batch_images = self.processor.process_images(images)

        # Move to device
        if self.device == "mps":
            batch_images = {k: v.to(self.device) for k, v in batch_images.items()}
        else:
            batch_images = batch_images.to(self.device)

        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(**batch_images)

        return embeddings

    def process_visual(self, file_path: str, file_type: str = None) -> Dict:
        """
        Process any visual file (PDF or image) with ColPali.

        Args:
            file_path: Path to file
            file_type: 'pdf' or 'image' (auto-detected if None)

        Returns:
            Dictionary with processing results
        """
        # Auto-detect type if not provided
        if file_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                file_type = 'pdf'
            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                file_type = 'image'
            else:
                return {
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'file_type': file_type,
                    'embeddings': None,
                    'processing_time': 0,
                    'device': self.device,
                    'error': f'Unsupported file type: {ext}'
                }

        # Process based on type
        if file_type == 'pdf':
            return self.process_pdf(file_path, show_progress=False)
        elif file_type == 'image':
            return self.process_image(file_path)
        else:
            return {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'file_type': file_type,
                'embeddings': None,
                'processing_time': 0,
                'device': self.device,
                'error': f'Unknown file type: {file_type}'
            }

    def process_image(self, image_path: str) -> Dict:
        """
        Process a standalone image file.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        if not os.path.exists(image_path):
            return {
                'image_path': image_path,
                'filename': os.path.basename(image_path),
                'embeddings': None,
                'processing_time': 0,
                'device': self.device,
                'error': 'Image not found'
            }

        try:
            image = Image.open(image_path)
            embeddings = self.generate_embeddings([image])
            processing_time = time.time() - start_time

            return {
                'image_path': image_path,
                'filename': os.path.basename(image_path),
                'embeddings': embeddings,
                'embedding_shape': tuple(embeddings.shape),
                'processing_time': processing_time,
                'device': self.device,
                'error': None
            }

        except Exception as e:
            return {
                'image_path': image_path,
                'filename': os.path.basename(image_path),
                'embeddings': None,
                'processing_time': time.time() - start_time,
                'device': self.device,
                'error': str(e)
            }

    def process_pdf(self, pdf_path: str, dpi: int = 200, show_progress: bool = True) -> Dict:
        """
        Process a PDF file end-to-end: convert to images and generate embeddings.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion
            show_progress: Whether to print progress info

        Returns:
            Dictionary with:
                - pdf_path: Original PDF path
                - filename: PDF filename
                - num_pages: Number of pages
                - embeddings: Tensor of embeddings
                - processing_time: Time taken (seconds)
                - device: Device used
                - error: Error message (if any)
        """
        start_time = time.time()

        # Convert PDF to images
        try:
            images = self.pdf_to_images(pdf_path, dpi=dpi)

            if not images:
                return {
                    'pdf_path': pdf_path,
                    'filename': os.path.basename(pdf_path),
                    'num_pages': 0,
                    'embeddings': None,
                    'processing_time': 0,
                    'device': self.device,
                    'error': 'Failed to convert PDF to images'
                }

            embeddings = self.generate_embeddings(images)
            processing_time = time.time() - start_time

            return {
                'pdf_path': pdf_path,
                'filename': os.path.basename(pdf_path),
                'num_pages': len(images),
                'embeddings': embeddings,
                'embedding_shape': tuple(embeddings.shape),
                'processing_time': processing_time,
                'time_per_page': processing_time / len(images),
                'device': self.device,
                'error': None
            }

        except Exception as e:
            return {
                'pdf_path': pdf_path,
                'filename': os.path.basename(pdf_path),
                'num_pages': 0,
                'embeddings': None,
                'processing_time': time.time() - start_time,
                'device': self.device,
                'error': str(e)
            }

    def generate_query_embedding(self, query: str) -> torch.Tensor:
        """
        Generate embedding for a text query.

        Args:
            query: Text query string

        Returns:
            Query embedding tensor
        """
        # Process query through ColPali processor
        batch_query = self.processor.process_queries([query])

        # Move to device
        if self.device == "mps":
            batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
        else:
            batch_query = batch_query.to(self.device)

        # Generate embedding
        with torch.no_grad():
            query_embedding = self.model(**batch_query)

        return query_embedding

    def compute_similarity(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> float:
        """
        Compute similarity score between query and document embeddings.

        Args:
            query_embedding: Query embedding tensor
            doc_embeddings: Document embeddings tensor

        Returns:
            Similarity score (higher is better)
        """
        # Use ColPali's multi-vector scoring
        scores = self.processor.score_multi_vector(query_embedding, doc_embeddings)
        # scores can be a tensor - get the scalar value
        if isinstance(scores, torch.Tensor):
            if scores.numel() == 1:
                return float(scores.item())
            else:
                # Multiple scores - take the first one
                return float(scores.flatten()[0].item())
        return float(scores[0]) if hasattr(scores, '__getitem__') else float(scores)


if __name__ == '__main__':
    print("ColPali Processor")
