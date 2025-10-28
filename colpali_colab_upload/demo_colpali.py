"""
ColPali RAG System Demo for VTA

Demonstrates the ColPali-based image retrieval system (PDFs and images).
Compare with demo.py which uses text-based PDF extraction.

NOTE: ColPali runs on CPU (slow but accurate). First run downloads ~2-3GB model.
      Processing PDFs takes 1-5 minutes each on CPU.
"""

import json
import os
from colpali_rag_system import ColPaliRAGSystem


def main():
    print("\n" + "="*70)
    print("  VTA ColPali RAG SYSTEM DEMO")
    print("  (Image-based Retrieval with ColQwen2)")
    print("="*70 + "\n")

    print("This demo uses ColQwen2 vision model for image content:")
    print("  Processes PDFs and images with visual understanding")
    print("  Captures tables, charts, diagrams, equations")
    print("  Slower processing (CPU-based, ~10-30 sec per item)")
    print("\nFor fast text-only retrieval, use demo.py instead.\n")

    # Initialize
    rag = ColPaliRAGSystem()

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
    print("Processing all image content (PDFs + images) automatically...\n")

    for post in posts:
        # Show what will be processed
        attachments = post.get('attachments', [])
        image_atts = [att for att in attachments if att.get('file_type') in ['pdf', 'image']]

        if image_atts:
            print(f"Post: {post['title']}")
            for att in image_atts:
                print(f"  - {att['file_type']}: {att['filename']}")

        result = rag.add_post(post, base_dir=posts_base_dir)
        if not result['success']:
            print(f"  Failed: {post['title']}")
        elif image_atts:
            pdf_count = result.get('pdf_count', 0)
            image_count = result.get('image_file_count', 0)
            total_pages = result.get('total_pages', 0)

            parts = []
            if pdf_count > 0:
                parts.append(f"{pdf_count} pdf ({total_pages} pages)" if pdf_count == 1 else f"{pdf_count} pdfs ({total_pages} pages)")
            if image_count > 0:
                parts.append(f"{image_count} image" if image_count == 1 else f"{image_count} images")

            print(f"  Done: {', '.join(parts)}\n")

    # Test queries
    print("\n--- STEP 2: Test Queries ---")

    queries = [
        "Where can we get the solution for assignment 2 question 5b",
        "When can we get assignment 2 solutions?",
        "Looking for final project group",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: \"{query}\"")
        print('='*70)

        results = rag.retrieve(query, n_results=5, similarity_threshold=0.5)

        # Show top match
        top = results['top_match']
        if top:
            similarity = top.get('similarity')
            below_threshold = top.get('below_threshold', False)

            if below_threshold:
                print(f"\n→ BEST MATCH (below threshold): POST - {top['title']}")
            else:
                print(f"\n→ TOP MATCH: POST - {top['title']}")

            print(f"  Author: {top.get('author', 'Unknown')}")
            if similarity is not None:
                print(f"  Similarity: {similarity:.4f} (0-1 scale, higher is better)")

            if top.get('matched_via') == 'image':
                file_type = top.get('matched_file_type', 'unknown')
                print(f"  Matched via {file_type}: {top['matched_image']}")
                print(f"    (ColQwen2 image embedding)")
            else:
                print(f"  Matched via post text")
                print(f"    (OpenAI text embedding)")

            # Show attachments in the post
            attachments = top.get('attachments', [])
            if attachments:
                print(f"\n  Attachments in post:")
                for att in attachments:
                    print(f"    • {att['filename']} ({att['file_type']})")

            if below_threshold:
                print(f"\n  Warning: Low confidence match - may not be relevant")
        else:
            print(f"\n→ No relevant information found in the database")

        # Show similar posts
        if results['similar_posts']:
            print(f"\n  Similar student posts:")
            for j, post in enumerate(results['similar_posts'][:3], 1):
                post_sim = post.get('similarity', 0)
                print(f"    {j}. {post['title']}")
                print(f"       by {post['author']} | similarity: {post_sim:.4f}")

        # Show course materials
        if results['course_materials']:
            print(f"\n  Related course materials (images):")
            for j, material in enumerate(results['course_materials'][:3], 1):
                mat_sim = material.get('similarity', 0)
                mat_type = material.get('file_type', 'unknown')
                print(f"    {j}. {material['filename']} ({mat_type})")
                print(f"       from: {material['parent_title']} | similarity: {mat_sim:.4f}")

    # Stats
    print("\n" + "="*70)
    print("--- STEP 3: Statistics ---")
    print("="*70)
    stats = rag.get_stats()
    print(f"Total text items: {stats['total_texts']}")
    print(f"Total image items (PDFs + images): {stats['total_images']}")
    print(f"Total items: {stats['total_items']}")


if __name__ == '__main__':
    main()
