"""
Main Evaluation — MMHal-Bench (Section 6.2, Benchmark 5).

Open-ended hallucination quality:
    - 96 images × 8 question types = 768 questions
    - Metric: Score 1-6 (GPT-4 evaluator)
    - Cost: ~$5 GPT-4 API

Usage:
    python experiments/main_eval/eval_mmhal.py \
        --config configs/default.yaml \
        --mmhal_path data/mmhal_bench/ \
        --gpt4_api_key $OPENAI_API_KEY
"""
