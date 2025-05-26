from collections import defaultdict, Counter
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="parthh01/llamagm-bongcloud", help="Dataset name")
    parser.add_argument("--max_context", type=int, default=64, help="Max context length")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load dataset and tokenizer
    dataset = load_dataset(args.dataset_name, split="train")  # or use a local script
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_context = args.max_context  # truncate context to this length

    # Frequency counts
    context_freq = Counter()
    context_next_token_freq = defaultdict(Counter)

    for item in tqdm(dataset, desc="Processing dataset"):
        prompt = item["prompt"]
        completion = item["completion"]
        full_text = prompt + completion

        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        for i in range(1, len(token_ids)):
            context = tuple(token_ids[max(0, i - max_context):i])
            next_token = token_ids[i]
            context_freq[context] += 1
            context_next_token_freq[context][next_token] += 1

    # Compute empirical conditional entropy
    total_weighted_entropy = 0
    total_context_occurrences = 0

    for context, next_token_counts in context_next_token_freq.items():
        total_contexts = sum(next_token_counts.values())
        entropy = 0.0
        for token_id, count in next_token_counts.items():
            p = count / total_contexts
            entropy -= p * math.log2(p)
        total_weighted_entropy += entropy * context_freq[context]
        total_context_occurrences += context_freq[context]

    expected_entropy_bits = total_weighted_entropy / total_context_occurrences
    expected_entropy_nats = expected_entropy_bits * math.log(2)  # convert to nats if desired

    print(f"Empirical conditional entropy (bits): {expected_entropy_bits:.4f}")
    print(f"Empirical conditional entropy (nats): {expected_entropy_nats:.4f}")
