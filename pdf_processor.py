"""
PDF processing module for RAG system.

Extracts text from PDFs and prepares them for embedding.
"""

import os
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import pymupdf4llm  # For markdown conversion
from cost_estimator import CostEstimator, estimate_pdf_cost


class PDFProcessor:
    """Process PDF files for RAG system."""

    def __init__(self):
        """Initialize PDF processor."""
        self.cost_estimator = CostEstimator()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file (plain text).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

        return text

    def extract_text_as_markdown(self, pdf_path: str) -> str:
        """
        Extract text from PDF as Markdown format.
        This preserves structure like headers, lists, tables, and formatting.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text in Markdown format
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            # Use pymupdf4llm to convert PDF to markdown
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            return markdown_text
        except Exception as e:
            print(f"Error extracting markdown from {pdf_path}: {e}")
            print(f"Falling back to plain text extraction...")
            return self.extract_text_from_pdf(pdf_path)

    def extract_text_by_page(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF, organized by page.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dicts with page number and text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text()
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'char_count': len(text)
                    })
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return []

        return pages

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text
            chunk_size: Target size for each chunk (in characters)
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the end
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:  # Only break if it's not too early
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def process_pdf_for_rag(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        overlap: int = 100,
        confirm_cost: bool = True
    ) -> Dict[str, any]:
        """
        Process a PDF file for RAG: extract text and create chunks.

        Args:
            pdf_path: Path to PDF file
            chunk_size: Target size for each chunk
            overlap: Overlap between chunks
            confirm_cost: If True, show cost estimate and ask for confirmation

        Returns:
            Dictionary with PDF metadata and chunks
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)

        if not text:
            return {
                'pdf_path': pdf_path,
                'filename': os.path.basename(pdf_path),
                'total_chars': 0,
                'chunks': [],
                'num_chunks': 0,
                'error': 'No text extracted'
            }

        # Estimate cost
        estimated_cost = self.cost_estimator.estimate_cost(text)
        estimated_tokens = self.cost_estimator.count_tokens(text)

        print(f"\n{'='*60}")
        print(f"COST ESTIMATE FOR: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        print(f"Text length: {len(text):,} characters")
        print(f"Estimated tokens: {estimated_tokens:,}")
        print(f"Estimated cost: ${estimated_cost:.6f}")
        print(f"{'='*60}\n")

        if confirm_cost:
            response = input("Proceed with processing? (y/n): ").strip().lower()
            if response != 'y':
                print("Processing cancelled.")
                return {
                    'pdf_path': pdf_path,
                    'filename': os.path.basename(pdf_path),
                    'total_chars': len(text),
                    'chunks': [],
                    'num_chunks': 0,
                    'error': 'User cancelled'
                }

        # Create chunks
        chunks = self.chunk_text(text, chunk_size, overlap)

        return {
            'pdf_path': pdf_path,
            'filename': os.path.basename(pdf_path),
            'total_chars': len(text),
            'chunks': chunks,
            'num_chunks': len(chunks),
            'chunk_size': chunk_size,
            'overlap': overlap
        }

    def process_multiple_pdfs(
        self,
        pdf_paths: List[str],
        chunk_size: int = 1000,
        overlap: int = 100,
        confirm_cost: bool = True
    ) -> List[Dict[str, any]]:
        """
        Process multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files
            chunk_size: Target size for each chunk
            overlap: Overlap between chunks

        Returns:
            List of processed PDF data
        """
        results = []

        for pdf_path in pdf_paths:
            print(f"Processing: {os.path.basename(pdf_path)}")
            result = self.process_pdf_for_rag(pdf_path, chunk_size, overlap, confirm_cost)
            results.append(result)

            if result.get('error'):
                print(f"  ⚠ {result['error']}")
            else:
                print(f"  ✓ Extracted {result['total_chars']} characters")
                print(f"  ✓ Created {result['num_chunks']} chunks")

        return results


def find_pdfs_in_directory(directory: str) -> List[str]:
    """
    Find all PDF files in a directory and subdirectories.

    Args:
        directory: Directory to search

    Returns:
        List of PDF file paths
    """
    pdf_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    return pdf_files


if __name__ == '__main__':
    print("PDF Processor - Use for extracting text from PDFs")
