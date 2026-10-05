"""Create a tiny, randomly initialized Qwen2-style model + tokenizer for smoke tests.

The real course uses Qwen/Qwen2.5-0.5B-Instruct on a GPU. This lets every notebook
run end to end on a CPU (it learns nothing useful, it only checks the code paths):

    uv run --group dev python tools/make_tiny_model.py /tmp/tiny-qwen
    RLCOURSE_MODEL=/tmp/tiny-qwen RLCOURSE_SMOKE=1 uv run --group dev python tools/build_notebooks.py --check
"""

import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

out = sys.argv[1]
special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
corpus = ["What is 12 + 34 * 5? Think step by step. <answer>182</answer> system user assistant You are"] * 50
tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=special,
                                                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
hf_tok.chat_template = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
cfg = Qwen2Config(vocab_size=len(hf_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                  eos_token_id=hf_tok.eos_token_id, pad_token_id=hf_tok.pad_token_id, tie_word_embeddings=True)
Qwen2ForCausalLM(cfg).save_pretrained(out)
hf_tok.save_pretrained(out)
print("saved", out, "vocab", len(hf_tok))
