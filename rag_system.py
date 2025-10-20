"""
Post-based RAG System for VTA

Posts and PDFs are embedded separately.
Always returns posts (includes PDF info when matched via PDF).
"""

import os
import json
from typing import List, Dict
import chromadb
from openai import OpenAI
from pdf_processor import PDFProcessor
from cost_estimator import CostEstimator
from config_loader import get_openai_key


class PostRAGSystem:
    """RAG system that retrieves posts based on relevance (posts and PDFs embedded separately)."""

    def __init__(self, collection_name: str = "ed_posts", persist_directory: str = "./rag_db", confirm_cost: bool = True):
        self.confirm_cost = confirm_cost
        self.pdf_processor = PDFProcessor()
        self.cost_estimator = CostEstimator()
        self.client = OpenAI(api_key=get_openai_key())

        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine distance instead of L2
        )

        print(f"✓ RAG System initialized ({self.collection.count()} items)")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI."""
        return self.client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

    def add_post(self, post_data: Dict, base_dir: str = None) -> Dict:
        """Add post to RAG system. Posts and PDFs embedded separately.

        Args:
            post_data: Post dictionary with content and attachments
            base_dir: Base directory for resolving relative paths (default: current directory)
        """
        post_id = post_data['post_id']
        title = post_data['title']
        content = post_data.get('content', [])
        attachments = post_data.get('attachments', [])

        # Check if post already exists
        try:
            existing = self.collection.get(ids=[f"post_{post_id}"])
            if existing and existing['ids']:
                print(f"\nSkipping: {title} (already exists)")
                return {'success': True, 'post_id': post_id, 'items_added': 0, 'total_cost': 0, 'skipped': True}
        except:
            pass

        print(f"\nProcessing: {title}")

        # Process post text
        post_text = f"Title: {title}\n\n" + "\n".join(content)

        # Always include attachments to improve semantic matching
        if attachments:
            attachment_names = [att.get('filename', '') for att in attachments if att.get('filename')]
            if attachment_names:
                post_text += "\n\nAttachments: " + ", ".join(attachment_names)

        # If content is empty, include comments for additional context
        if not content or len("\n".join(content).strip()) == 0:
            comments = post_data.get('comments', [])
            if comments:
                comment_texts = []
                for comment in comments[:3]:  # Include up to 3 top comments
                    comment_texts.append(comment.get('comment_text', ''))
                if comment_texts:
                    post_text += "\n\nComments:\n" + "\n".join(comment_texts)

        post_cost = self.cost_estimator.estimate_cost(post_text)

        # Process PDFs
        pdf_data = []
        pdf_costs = []
        for i, att in enumerate(attachments):
            if att.get('file_type') == 'pdf' and att.get('local_path'):
                # Resolve path (handle both relative and absolute paths)
                local_path = att['local_path']
                if not os.path.isabs(local_path) and base_dir:
                    local_path = os.path.join(base_dir, local_path)

                if os.path.exists(local_path):
                    pdf_text = self.pdf_processor.extract_text_as_markdown(local_path)
                    if pdf_text:
                        pdf_costs.append(self.cost_estimator.estimate_cost(pdf_text))
                        pdf_data.append({'index': i, 'filename': att['filename'], 'text': pdf_text, 'attachment': att})
                else:
                    print(f"  Warning: PDF not found: {local_path}")

        total_cost = post_cost + sum(pdf_costs)
        print(f"Cost: ${total_cost:.6f} (Post: ${post_cost:.6f}, {len(pdf_data)} PDFs)")

        if self.confirm_cost:
            if input("Proceed? (y/n): ").strip().lower() != 'y':
                return {'success': False, 'error': 'Cancelled'}

        # Generate embeddings and store
        items = []

        # Add post
        self.cost_estimator.add_usage(post_text)
        post_doc = json.dumps({
            'type': 'post', 'post_id': post_id, 'title': title, 'content': content,
            'author': post_data.get('author'), 'author_role': post_data.get('author_role'),
            'timestamp': post_data.get('timestamp'), 'post_url': post_data.get('post_url'),
            'attachments': attachments, 'comments': post_data.get('comments', [])
        })
        items.append((f"post_{post_id}", self.generate_embedding(post_text), post_doc,
                     {'type': 'post', 'post_id': post_id, 'title': title}))

        # Add PDFs
        for pdf in pdf_data:
            self.cost_estimator.add_usage(pdf['text'])
            pdf_doc = json.dumps({
                'type': 'pdf', 'post_id': post_id, 'filename': pdf['filename'],
                'content': pdf['text'], 'attachment': pdf['attachment'],
                'parent_post': {'title': title, 'content': content, 'author': post_data.get('author'),
                               'post_url': post_data.get('post_url'), 'timestamp': post_data.get('timestamp')}
            })
            items.append((f"pdf_{post_id}_{pdf['index']}", self.generate_embedding(pdf['text']), pdf_doc,
                         {'type': 'pdf', 'post_id': post_id, 'filename': pdf['filename']}))

        self.collection.add(
            ids=[item[0] for item in items],
            embeddings=[item[1] for item in items],
            documents=[item[2] for item in items],
            metadatas=[item[3] for item in items]
        )

        print(f"✓ Added {len(items)} items ({self.collection.count()} total)")
        return {'success': True, 'post_id': post_id, 'items_added': len(items), 'total_cost': total_cost}

    def retrieve(self, question: str, n_results: int = 3, include_similar_posts: bool = True, similarity_threshold: float = 0.5) -> Dict:
        """
        Retrieve most relevant posts for a question.

        Args:
            question: Student's question
            n_results: Number of results to return
            include_similar_posts: If True, also includes similar student posts beyond the top match
            similarity_threshold: Minimum similarity score [0,1] for valid match (default 0.5, higher = stricter)

        Returns:
            Dictionary with:
            - 'top_match': Most relevant post (always a post, includes 'matched_pdf' field when matched via PDF)
            - 'similar_posts': Other relevant posts (if include_similar_posts=True)
            - 'course_materials': Relevant PDFs from course materials
        """
        results = self.collection.query(query_embeddings=[self.generate_embedding(question)], n_results=n_results)

        items = []
        for i in range(len(results['documents'][0])):
            item_data = json.loads(results['documents'][0][i])
            distance = results['distances'][0][i] if 'distances' in results else None
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            # So: cosine_similarity = 1 - distance
            similarity = (1 - distance) if distance is not None else None
            item_data['distance'] = distance
            item_data['similarity'] = similarity
            items.append(item_data)

        # Get top match - if it's a PDF, convert to parent post
        top_match = None
        matched_via_pdf = False
        below_threshold = False

        if items:
            # Check if top match meets similarity threshold
            top_similarity = items[0].get('similarity')
            if top_similarity is not None and top_similarity < similarity_threshold:
                below_threshold = True
                print(f"✗ No match above threshold (best: {top_similarity:.4f} < threshold: {similarity_threshold})")

            # Always return the best match, but mark if below threshold
            if items[0]['type'] == 'pdf':
                # PDF matched - return parent post with PDF info
                matched_via_pdf = True
                parent = items[0]['parent_post']
                top_match = {
                    'type': 'post',
                    'post_id': items[0]['post_id'],
                    'title': parent['title'],
                    'content': parent['content'],
                    'author': parent['author'],
                    'post_url': parent['post_url'],
                    'timestamp': parent.get('timestamp'),
                    'matched_pdf': items[0]['filename'],
                    'matched_via': 'pdf',
                    'distance': items[0].get('distance'),
                    'similarity': items[0].get('similarity'),
                    'below_threshold': below_threshold
                }
            else:
                # Post matched directly
                top_match = items[0]
                top_match['matched_via'] = 'post'
                top_match['below_threshold'] = below_threshold

        # Organize remaining results - filter by similarity threshold
        similar_posts = [item for item in items[1:] if item['type'] == 'post' and item.get('similarity', 0) >= similarity_threshold]
        course_materials = [item for item in items if item['type'] == 'pdf' and item.get('similarity', 0) >= similarity_threshold]

        if top_match:
            if matched_via_pdf:
                print(f"✓ Top match: POST (via PDF) - {top_match['title']}")
                print(f"  Matched PDF: {top_match['matched_pdf']}")
            else:
                print(f"✓ Top match: POST - {top_match['title']}")

            if len(similar_posts) > 0:
                print(f"  + {len(similar_posts)} similar student post(s)")
            if len(course_materials) > 0:
                print(f"  + {len(course_materials)} course material(s)")

        return {
            'question': question,
            'top_match': top_match,
            'similar_posts': similar_posts,
            'course_materials': course_materials,
            'all_items': items
        }

    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            'total_items': self.collection.count(),
            'total_cost': self.cost_estimator.get_total_cost(),
            'total_tokens': self.cost_estimator.total_tokens
        }
