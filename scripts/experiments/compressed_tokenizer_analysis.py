#!/usr/bin/env python3
"""
Analyze potential Prompt Lookup improvement from tokenizer compression.

Tests whether Engram-style token normalization could increase N-gram matches
in Prompt Lookup speculation.

Source: Engram paper (DeepSeek, Jan 2025)
"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs"
"""

import re
import unicodedata
from collections import defaultdict
from transformers import AutoTokenizer

def compress_token(text: str) -> str:
    """Engram-style token compression (from paper's CompressedTokenizer)."""
    if not text:
        return "<empty>"
    # NFKC + NFD + strip accents
    text = unicodedata.normalize('NFKC', text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Lowercase + collapse whitespace
    text = text.lower()
    text = re.sub(r'[ \t\r\n]+', ' ', text).strip()
    return text if text else "<empty>"

def build_ngram_index(token_ids: list, tokenizer, n: int, compress: bool = False):
    """Build N-gram → positions index."""
    index = defaultdict(list)

    for i in range(len(token_ids) - n + 1):
        ngram_ids = tuple(token_ids[i:i+n])

        if compress:
            # Normalize token text before indexing
            texts = [tokenizer.decode([tid]) for tid in ngram_ids]
            key = tuple(compress_token(t) for t in texts)
        else:
            key = ngram_ids

        index[key].append(i)

    return index

def analyze_prompt(prompt_text: str, model_name: str, label: str = ""):
    """Analyze N-gram repetition with and without compression."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    token_ids = tokenizer.encode(prompt_text)

    print(f"\n{'=' * 70}")
    print(f"TEST: {label}")
    print(f"{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"Tokens: {len(token_ids)}, Text: {len(prompt_text)} chars")

    results = {}
    for n in [3, 4, 5]:  # Focus on N-grams typically used by Prompt Lookup
        raw_index = build_ngram_index(token_ids, tokenizer, n, compress=False)
        compressed_index = build_ngram_index(token_ids, tokenizer, n, compress=True)

        raw_unique = len(raw_index)
        compressed_unique = len(compressed_index)
        reduction = (raw_unique - compressed_unique) / raw_unique * 100 if raw_unique > 0 else 0

        # Count multi-occurrence ngrams (these are the ones Prompt Lookup can exploit)
        raw_repeats = sum(1 for positions in raw_index.values() if len(positions) > 1)
        compressed_repeats = sum(1 for positions in compressed_index.values() if len(positions) > 1)

        # Total repeat instances (positions that could be speculated)
        raw_repeat_positions = sum(len(p) - 1 for p in raw_index.values() if len(p) > 1)
        compressed_repeat_positions = sum(len(p) - 1 for p in compressed_index.values() if len(p) > 1)

        improvement = ((compressed_repeat_positions - raw_repeat_positions) / raw_repeat_positions * 100
                      if raw_repeat_positions > 0 else 0)

        results[n] = {
            'raw_unique': raw_unique,
            'compressed_unique': compressed_unique,
            'reduction_pct': reduction,
            'raw_repeats': raw_repeats,
            'compressed_repeats': compressed_repeats,
            'raw_positions': raw_repeat_positions,
            'compressed_positions': compressed_repeat_positions,
            'improvement_pct': improvement,
        }

        delta_repeats = compressed_repeats - raw_repeats
        delta_positions = compressed_repeat_positions - raw_repeat_positions

        print(f"\n  {n}-gram:")
        print(f"    Unique N-grams:      {raw_unique:5d} → {compressed_unique:5d} ({reduction:5.1f}% vocab reduction)")
        print(f"    Repeating N-grams:   {raw_repeats:5d} → {compressed_repeats:5d} ({delta_repeats:+4d})")
        print(f"    Speculatable pos:    {raw_repeat_positions:5d} → {compressed_repeat_positions:5d} ({delta_positions:+4d}, {improvement:+5.1f}%)")

    return results

def main():
    # Use Qwen tokenizer (same family as our production models)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    all_results = {}

    # Test 1: Document with intentional case/formatting variations
    test_prompt_1 = '''
    Summarize the following document about Alexander the Great:

    Alexander the Great (356-323 BC) was a king of Macedonia. Alexander's
    military genius led him to conquer the Persian Empire. ALEXANDER
    defeated Darius III at the Battle of Gaugamela. The legacy of alexander
    the great influenced Hellenistic culture for centuries.

    Key facts about Alexander:
    - Alexander was tutored by Aristotle
    - alexander conquered Egypt and founded Alexandria
    - ALEXANDER THE GREAT died at age 32

    In conclusion, Alexander the Great remains one of history's most
    significant military commanders. Alexander's empire stretched from
    Greece to India.
    '''
    all_results['case_variations'] = analyze_prompt(test_prompt_1, model_name,
        "Case variations (artificial worst-case)")

    # Test 2: Code with variable naming patterns
    test_prompt_2 = '''
    Review and refactor this code:

    ```python
    def getUserData(userId):
        user_data = fetch_user(userId)
        userData = process(user_data)
        return userData

    def get_user_data(user_id):
        userData = fetch_user(user_id)
        user_data = process(userData)
        return user_data

    class UserDataProcessor:
        def process_user_data(self, user_data):
            processed = self.validate(user_data)
            return processed
    ```

    The getUserData and get_user_data functions should be consolidated.
    '''
    all_results['code_naming'] = analyze_prompt(test_prompt_2, model_name,
        "Code with naming variations")

    # Test 3: Real summarization prompt (typical Prompt Lookup use case - exact repeats)
    test_prompt_3 = '''
    The transformer architecture was introduced in "Attention Is All You Need".
    The transformer uses self-attention mechanisms. The attention mechanism
    allows the transformer to process sequences in parallel. The transformer
    architecture has become the foundation for modern NLP. The attention
    mechanism in the transformer computes query, key, and value projections.
    The transformer model processes input through multiple layers. Each layer
    in the transformer contains attention and feed-forward components.
    The transformer architecture revolutionized natural language processing.
    '''
    all_results['exact_repeats'] = analyze_prompt(test_prompt_3, model_name,
        "Exact phrase repetition (typical summarization)")

    # Test 4: Real-world README/documentation (mixed case in headings)
    test_prompt_4 = '''
    # Installation Guide

    ## Installation

    To complete the installation, follow these steps:

    1. Run the installation script
    2. Verify the installation completed successfully
    3. Test the installation by running the test suite

    ### Troubleshooting Installation Issues

    If installation fails, check the installation logs.
    Common installation problems include:
    - Missing dependencies during installation
    - Permission errors in installation directory

    ## Configuration

    After installation, configure the system settings.
    '''
    all_results['documentation'] = analyze_prompt(test_prompt_4, model_name,
        "Documentation with heading variations")

    # Test 5: Long technical document (realistic)
    test_prompt_5 = '''
    The Model Context Protocol (MCP) provides a standardized way for AI
    applications to connect with external data sources and tools. MCP
    enables seamless integration between AI models and various services.

    The protocol defines three core primitives:
    - Resources: Data exposed by MCP servers
    - Tools: Functions that can be called via MCP
    - Prompts: Templates for common MCP interactions

    MCP servers expose resources that MCP clients can access. When an MCP
    client connects to an MCP server, it can discover available resources
    and tools. The MCP specification ensures interoperability between
    different MCP implementations.

    To implement an MCP server:
    1. Define the resources your MCP server will expose
    2. Implement the MCP protocol handlers
    3. Register tools with the MCP runtime

    MCP clients connect to MCP servers using the MCP transport layer.
    '''
    all_results['technical_doc'] = analyze_prompt(test_prompt_5, model_name,
        "Technical documentation (realistic)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Improvement from Compression")
    print("=" * 70)
    print(f"\n{'Test Case':<35} {'3-gram':>10} {'4-gram':>10} {'5-gram':>10}")
    print("-" * 70)

    for test_name, results in all_results.items():
        improvements = [f"{results[n]['improvement_pct']:+.1f}%" for n in [3, 4, 5]]
        print(f"{test_name:<35} {improvements[0]:>10} {improvements[1]:>10} {improvements[2]:>10}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

if __name__ == "__main__":
    main()
