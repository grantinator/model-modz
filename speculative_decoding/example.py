"""
References:
- https://jaykmody.com/blog/speculative-sampling/

todo:
- [done] Generate output with llama-cpp-python
- [wip] Extract logits for k generated tokens
- [] Implement token acceptance logic (and understand it...)
- [] Test speculative decoding parity / speedup vs. using target model only
"""

import os

from huggingface_hub import login
from llama_cpp import Llama, llama_get_logits, llama_n_vocab, llama_token_data
import numpy as np


HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN is not None, "Set $HF_TOKEN in environment."
login(HF_TOKEN)


def generate_with_speculator(
    prompt: str,
    speculator_model: Llama,
    target_model: Llama,
    k: int = 5,
    max_tokens: int = 32,
) -> str:
    """Generates text using a speculator and target model."""
    tokens = target_model.tokenizer().tokenize(prompt.encode())
    num_input_tokens = len(tokens)
    while len(tokens) < num_input_tokens + max_tokens:
        # Speculate the next k tokens.
        # temp=0 to use greedy (i.e. deterministic) sampling
        sample = speculator_model.generate(tokens=tokens, temp=0.0)
        draft_tokens = []
        for _ in range(k):
            draft_tokens.append(next(sample))

        draft_token_logits = speculator_model.scores[
            len(tokens) - 1 : len(tokens) + k - 1, :
        ]
        assert np.all(
            np.equal(np.argmax(draft_token_logits, axis=1), np.array(draft_tokens))
        )

        # Score the k draft tokens with the big model.
        all_accepted = True
        target_model.eval(tokens + draft_tokens)

        # Note: We include the score for the next token, too.
        scored_token_logits = target_model.scores[len(tokens) - 1 : len(tokens) + k, :]
        for i in range(k):
            # Index of draft token
            j = draft_tokens[i]

            if np.random.random() < min(
                1, scored_token_logits[i][j] / draft_token_logits[i][j]
            ):
                # Accept token if target model probability is ~higher~ than draft model probability.
                tokens.append(j)
            else:
                # todo: in the paper, they are sampling based on both target and draft distributions not this...idk why
                tokens.append(np.argmax(scored_token_logits[i]))
                all_accepted = False
                break

        if all_accepted:
            # sample again from target model to take advantage of the fact
            # that we already produced logits for the next (n + k)th token
            tokens.append(np.argmax(scored_token_logits[-1]))

    return tokens


if __name__ == "__main__":
    # Note: llama.cpp is the fastest way I've found to run LLMs on Macbooks.
    speculator_llm = Llama.from_pretrained(
        repo_id="TheBloke/Llama-2-7B-GGUF",
        # ~2.83 GB, "smallest, significant quality loss - not recommended for most purposes"
        filename="llama-2-7b.Q2_K.gguf",
        # This is needed to return the logits for all tokens during generation.
        logits_all=True,
    )
    target_llm = Llama.from_pretrained(
        repo_id="TheBloke/Llama-2-7B-GGUF",
        # ~4.08 GB, "medium, balanced quality - recommended"
        filename="llama-2-7b.Q4_K_M.gguf",
        # This is needed to return the logits for all tokens during generation.
        logits_all=True,
    )

    tokens = generate_with_speculator(
        prompt="Q: Name the planets in the solar system? A: ",
        speculator_model=speculator_llm,
        target_model=target_llm,
    )
    print(target_llm.tokenizer().decode(tokens))

    # output = llm(
    #     "Q: Name the planets in the solar system? A: ",
    #     max_tokens=32,
    #     stop=["Q:", "\n"], # stop generating just before the model would generate a new question
    #     echo=True, # echo the prompt back in the output
    #     logprobs=True, # return logits for all tokens in the prompt + output
    # )
