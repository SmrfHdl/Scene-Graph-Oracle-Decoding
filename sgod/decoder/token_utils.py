"""
Token utilities for the decoder.

Handles BPE tokenizer issues (Risk 2, Section 5):
- word_to_token_ids(word, tokenizer) -> List[int]
- Top-K candidate extraction and re-ranking (Risk 3)
- EOS detection
"""
