"""
Cost estimation and safety limits for OpenAI API usage.

Helps track and limit spending on embeddings and LLM calls.
"""

import tiktoken


class CostEstimator:
    """Estimate and track OpenAI API costs for text-embedding-3-small."""

    # Pricing: text-embedding-3-small @ $0.02 per 1M tokens
    PRICE_PER_MILLION_TOKENS = 0.02

    def __init__(self):
        """Initialize cost estimator for text-embedding-3-small."""
        self.model = 'text-embedding-3-small'
        self.total_tokens = 0
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text):
        """
        Count tokens in text.

        Args:
            text: Input text string

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def estimate_cost(self, text):
        """
        Estimate cost for processing text.

        Args:
            text: Input text string

        Returns:
            Estimated cost in USD
        """
        tokens = self.count_tokens(text)
        cost_per_token = self.PRICE_PER_MILLION_TOKENS / 1_000_000
        return tokens * cost_per_token

    def add_usage(self, text):
        """
        Track token usage.

        Args:
            text: Text that was processed

        Returns:
            Tokens used for this text
        """
        tokens = self.count_tokens(text)
        self.total_tokens += tokens
        return tokens

    def get_total_cost(self):
        """
        Get total cost so far.

        Returns:
            Total cost in USD
        """
        cost_per_token = self.PRICE_PER_MILLION_TOKENS / 1_000_000
        return self.total_tokens * cost_per_token

    def get_summary(self):
        """
        Get usage summary.

        Returns:
            Dictionary with usage stats
        """
        return {
            'total_tokens': self.total_tokens,
            'total_cost_usd': self.get_total_cost(),
            'model': self.model
        }

    def print_summary(self):
        """Print usage summary."""
        summary = self.get_summary()
        print(f"\nCost Summary: {summary['total_tokens']:,} tokens, ${summary['total_cost_usd']:.6f}")


def estimate_pdf_cost(num_pages, model='text-embedding-3-small'):
    """
    Estimate cost for embedding a PDF.

    Args:
        num_pages: Number of pages in PDF
        model: Embedding model to use

    Returns:
        Dictionary with cost estimates
    """
    # Rough estimates
    tokens_per_page = 600  # Average
    total_tokens = num_pages * tokens_per_page

    cost_per_token = CostEstimator.PRICING.get(model, 0) / 1_000_000
    total_cost = total_tokens * cost_per_token

    return {
        'num_pages': num_pages,
        'estimated_tokens': total_tokens,
        'estimated_cost_usd': total_cost,
        'model': model
    }


def check_cost_limit(estimated_cost, limit=1.0):
    """
    Check if estimated cost exceeds limit.

    Args:
        estimated_cost: Estimated cost in USD
        limit: Maximum allowed cost in USD

    Returns:
        True if within limit, False otherwise
    """
    if estimated_cost > limit:
        print(f"⚠️  WARNING: Estimated cost ${estimated_cost:.4f} exceeds limit ${limit:.2f}")
        return False
    return True


def print_cost_examples():
    """Print example costs for reference."""
    print("\n" + "="*70)
    print("COST EXAMPLES (text-embedding-3-small @ $0.02 per 1M tokens)")
    print("="*70)

    examples = [
        (1, "1-page document"),
        (5, "5-page PDF"),
        (10, "10-page PDF"),
        (50, "50-page textbook chapter"),
        (100, "100-page document"),
        (500, "500-page textbook"),
    ]

    for pages, description in examples:
        est = estimate_pdf_cost(pages)
        print(f"{description:30s} ~ {est['estimated_tokens']:6,} tokens = ${est['estimated_cost_usd']:.6f}")

    print("="*70)
    print("Note: These are rough estimates. Actual costs may vary.")
    print("="*70 + "\n")


if __name__ == '__main__':
    print_cost_examples()
