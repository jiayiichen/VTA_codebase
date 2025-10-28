"""
ColPali-based RAG System for VTA

Uses ColQwen2 for image retrieval (PDFs and images) and OpenAI embeddings for text.

NOTE: ColQwen2 runs on CPU (MPS not fully supported). Processing is slower than text-based
approach but captures visual elements like tables, charts, and diagrams.
"""

import os
import json
import pickle
from typing import List, Dict
from openai import OpenAI
from colpali_processor import ColPaliProcessor
from config_loader import get_openai_key
import torch


class ColPaliRAGSystem:
    """Hybrid RAG system: OpenAI embeddings for text, ColPali for images."""

    def __init__(self, persist_directory: str = "./colpali_rag_db"):
        """
        Initialize ColPali RAG system.

        Args:
            persist_directory: Directory to store embeddings
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize OpenAI for text embeddings (post content)
        self.openai_client = OpenAI(api_key=get_openai_key())

        # Initialize ColPali for image embeddings (PDFs and images)
        self.colpali_processor = ColPaliProcessor()

        # Storage
        self.texts = {}  # Text embeddings (OpenAI)
        self.images = {}  # Image embeddings (ColQwen2 - PDFs and images)

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing embeddings from disk."""
        texts_file = os.path.join(self.persist_directory, "texts.pkl")
        images_file = os.path.join(self.persist_directory, "images.pkl")

        if os.path.exists(texts_file):
            with open(texts_file, 'rb') as f:
                self.texts = pickle.load(f)

        if os.path.exists(images_file):
            with open(images_file, 'rb') as f:
                self.images = pickle.load(f)

    def _save_data(self):
        """Save embeddings to disk."""
        texts_file = os.path.join(self.persist_directory, "texts.pkl")
        images_file = os.path.join(self.persist_directory, "images.pkl")

        with open(texts_file, 'wb') as f:
            pickle.dump(self.texts, f)

        with open(images_file, 'wb') as f:
            pickle.dump(self.images, f)

    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text."""
        return self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

    def add_post(self, post_data: Dict, base_dir: str = None) -> Dict:
        """
        Add post to RAG system.
        - Text: OpenAI text embeddings
        - Images: ColPali image embeddings (PDFs and images)

        Args:
            post_data: Post dictionary with content and attachments
            base_dir: Base directory for resolving relative paths

        Returns:
            Dictionary with processing results
        """
        post_id = post_data['post_id']
        title = post_data['title']
        content = post_data.get('content', [])
        attachments = post_data.get('attachments', [])

        # Check if already processed
        if f"text_{post_id}" in self.texts:
            return {'success': True, 'post_id': post_id, 'items_added': 0, 'skipped': True}

        # Process post text with OpenAI
        post_text = f"Title: {title}\n\n" + "\n".join(content)

        if attachments:
            attachment_names = [att.get('filename', '') for att in attachments if att.get('filename')]
            if attachment_names:
                post_text += "\n\nAttachments: " + ", ".join(attachment_names)

        if not content or len("\n".join(content).strip()) == 0:
            comments = post_data.get('comments', [])
            if comments:
                comment_texts = [comment.get('comment_text', '') for comment in comments[:3]]
                if comment_texts:
                    post_text += "\n\nComments:\n" + "\n".join(comment_texts)

        text_embedding = self.generate_text_embedding(post_text)

        self.texts[f"text_{post_id}"] = {
            'post_id': post_id,
            'title': title,
            'content': content,
            'author': post_data.get('author'),
            'author_role': post_data.get('author_role'),
            'timestamp': post_data.get('timestamp'),
            'post_url': post_data.get('post_url'),
            'attachments': attachments,
            'comments': post_data.get('comments', []),
            'text_embedding': text_embedding
        }

        # Process image attachments (PDFs and images) with ColPali
        image_count = 0
        total_pages = 0
        pdf_count = 0
        image_file_count = 0

        for i, att in enumerate(attachments):
            if att.get('file_type') in ['pdf', 'image'] and att.get('local_path'):
                local_path = att['local_path']
                if not os.path.isabs(local_path) and base_dir:
                    local_path = os.path.join(base_dir, local_path)

                if os.path.exists(local_path):
                    # Process with auto-detection
                    result = self.colpali_processor.process_visual(local_path)

                    if result['error'] is None:
                        file_type = att.get('file_type', 'unknown')
                        image_id = f"{file_type}_{post_id}_{i}"
                        num_pages = result.get('num_pages', 1)

                        self.images[image_id] = {
                            'image_id': image_id,
                            'post_id': post_id,
                            'filename': att['filename'],
                            'file_type': file_type,
                            'attachment': att,
                            'parent_post': {
                                'title': title,
                                'content': content,
                                'author': post_data.get('author'),
                                'post_url': post_data.get('post_url'),
                                'timestamp': post_data.get('timestamp')
                            },
                            'colpali_embeddings': result['embeddings'].cpu(),
                            'num_pages': num_pages,
                            'processing_time': result['processing_time']
                        }
                        image_count += 1
                        total_pages += num_pages

                        if file_type == 'pdf':
                            pdf_count += 1
                        else:
                            image_file_count += 1
                    else:
                        print(f"  Warning: Failed to process {att['filename']}: {result['error']}")
                else:
                    print(f"  Warning: File not found: {local_path}")

        self._save_data()

        return {
            'success': True,
            'post_id': post_id,
            'items_added': 1 + image_count,
            'texts': 1,
            'images': image_count,
            'total_pages': total_pages,
            'pdf_count': pdf_count,
            'image_file_count': image_file_count
        }

    def retrieve(self, question: str, n_results: int = 3, similarity_threshold: float = 0.5) -> Dict:
        """
        Retrieve relevant text and image content for a question.

        Args:
            question: Student's question
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score [0,1]

        Returns:
            Dictionary with top_match, similar_posts, and course_materials
        """
        # Generate query embeddings
        query_text_emb = torch.tensor(self.generate_text_embedding(question))
        query_image_emb = self.colpali_processor.generate_query_embedding(question)

        # Score all text content
        text_scores = []
        for text_id, text_data in self.texts.items():
            text_emb = torch.tensor(text_data['text_embedding'])
            similarity = torch.nn.functional.cosine_similarity(
                query_text_emb.unsqueeze(0),
                text_emb.unsqueeze(0)
            ).item()
            text_scores.append((similarity, text_id, text_data, 'text'))

        # Score all image content
        image_scores = []
        for image_id, image_data in self.images.items():
            image_emb = image_data['colpali_embeddings']
            similarity = self.colpali_processor.compute_similarity(query_image_emb, image_emb)
            similarity_normalized = min(similarity / 100, 1.0)
            image_scores.append((similarity_normalized, image_id, image_data, 'image'))

        # Combine and sort
        all_scores = text_scores + image_scores
        all_scores.sort(key=lambda x: x[0], reverse=True)

        # Get top match
        top_match = None
        if all_scores:
            top_sim, top_id, top_data, top_type = all_scores[0]

            if top_sim >= similarity_threshold:
                if top_type == 'image':
                    parent = top_data['parent_post']
                    top_match = {
                        'type': 'post',
                        'post_id': top_data['post_id'],
                        'title': parent['title'],
                        'content': parent['content'],
                        'author': parent['author'],
                        'post_url': parent['post_url'],
                        'timestamp': parent.get('timestamp'),
                        'matched_image': top_data['filename'],
                        'matched_file_type': top_data.get('file_type'),
                        'matched_via': 'image',
                        'similarity': top_sim,
                        'below_threshold': False
                    }
                else:
                    top_match = {
                        'type': 'post',
                        'post_id': top_data['post_id'],
                        'title': top_data['title'],
                        'content': top_data['content'],
                        'author': top_data['author'],
                        'post_url': top_data['post_url'],
                        'timestamp': top_data.get('timestamp'),
                        'matched_via': 'text',
                        'similarity': top_sim,
                        'below_threshold': False,
                        'attachments': top_data.get('attachments', [])
                    }

        # Get similar posts (from text matches)
        similar_posts = [
            {
                'type': 'post',
                'post_id': data['post_id'],
                'title': data['title'],
                'content': data['content'],
                'author': data['author'],
                'similarity': sim
            }
            for sim, _, data, typ in all_scores[1:n_results+1]
            if typ == 'text' and sim >= similarity_threshold
        ]

        # Get course materials (from image matches)
        course_materials = [
            {
                'type': 'image',
                'filename': data['filename'],
                'file_type': data.get('file_type'),
                'post_id': data['post_id'],
                'parent_title': data['parent_post']['title'],
                'similarity': sim
            }
            for sim, _, data, typ in all_scores
            if typ == 'image' and sim >= similarity_threshold
        ][:n_results]

        return {
            'question': question,
            'top_match': top_match,
            'similar_posts': similar_posts,
            'course_materials': course_materials
        }

    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            'total_texts': len(self.texts),
            'total_images': len(self.images),
            'total_items': len(self.texts) + len(self.images)
        }


if __name__ == '__main__':
    print("ColPali RAG System")
