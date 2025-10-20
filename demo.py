"""
RAG System Demo for VTA

Demonstrates:
1. Loading posts from posts/dashboard.json
2. Adding posts with PDFs (embedded separately)
3. Retrieving relevant items based on student questions
4. Always returning posts (includes matched_pdf field when matched via PDF)
"""

import json
import os
from rag_system import PostRAGSystem

def main():
    print("\n" + "="*70)
    print("  VTA RAG SYSTEM DEMO")
    print("="*70 + "\n")

    # Initialize
    rag = PostRAGSystem(confirm_cost=True)

    # Load posts from posts folder
    posts_file = os.path.join(os.path.dirname(__file__), 'posts', 'dashboard.json')
    posts_base_dir = os.path.join(os.path.dirname(__file__), 'posts')

    print(f"Loading posts from: {posts_file}")

    with open(posts_file, 'r') as f:
        data = json.load(f)

    posts = data['posts']
    print(f"Found {len(posts)} posts\n")

    # Add all posts
    print("--- STEP 1: Add Posts ---")
    for post in posts:
        result = rag.add_post(post, base_dir=posts_base_dir)

    # Test queries
    print("\n--- STEP 2: Test Queries ---")

    queries = [
        "Where can we get the solution for assignment 2 question 5b",
        "When can we get assignment 2 solutions?",
        "Looking for final project group",
        
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        results = rag.retrieve(query, n_results=5, similarity_threshold=0.5)

        # Show top match
        top = results['top_match']
        if top:
            similarity = top.get('similarity')
            below_threshold = top.get('below_threshold', False)

            if below_threshold:
                print(f"   → BEST MATCH (below threshold): POST - {top['title']}")
            else:
                print(f"   → TOP: POST - {top['title']}")

            print(f"     Author: {top.get('author', 'Unknown')}")
            if similarity is not None:
                print(f"     Similarity: {similarity:.4f} (0-1 scale, higher is better)")
            if top.get('matched_via') == 'pdf':
                print(f"     (Matched via PDF: {top['matched_pdf']})")

            # Show attachments in the post
            attachments = top.get('attachments', [])
            if attachments:
                print(f"     Attachments: {len(attachments)} file(s)")
                for att in attachments:
                    print(f"       - {att['filename']} ({att['file_type']})")

            if below_threshold:
                print(f"     ⚠️  Warning: Low confidence match - may not be relevant")
        else:
            print(f"   → No relevant information found in the database")

        # Show similar posts
        if results['similar_posts']:
            print(f"\n   Similar student posts:")
            for post in results['similar_posts'][:3]:
                post_sim = post.get('similarity', 0)
                print(f"     • {post['title']} (by {post['author']}, similarity: {post_sim:.4f})")

        # Show course materials
        if results['course_materials']:
            print(f"\n   Related course materials:")
            for pdf in results['course_materials'][:3]:
                pdf_sim = pdf.get('similarity', 0)
                print(f"     • {pdf['filename']} (similarity: {pdf_sim:.4f})")

    # Stats
    print("\n--- STEP 3: Statistics ---")
    stats = rag.get_stats()
    print(f"Total items: {stats['total_items']}")
    print(f"Total cost: ${stats['total_cost']:.6f}")

    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()
