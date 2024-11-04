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
    predicted_tokens = []
    context = prompt
    while len(predicted_tokens) < max_tokens:
        # Speculate the next k tokens.
        speculated_output = speculator_model(
            context, max_tokens=k, logprobs=True, echo=True
        )
        speculated_text = speculated_output["choices"][0]["text"]
        speculated_logprobs = speculated_output["choices"][0]["logprobs"]

        # Okay it turns out we need logits for all tokens in vocab, not just token with max logit (i.e. the one that was generated).
        # So this line isn't useful but keeping here for reference.
        speculated_last_k_logprobs = speculated_logprobs["token_logprobs"][
            len(speculated_logprobs["tokens"]) - k :
        ]

        # Extract logits for all tokens. We have to use the low-level ctypes API :(.
        logits = llama_get_logits(speculator_model.ctx)
        n_vocab = llama_n_vocab(speculator_model.model)

        # (num tokens in context, n_vocab)
        arr = (llama_token_data * n_vocab)(
            *[
                llama_token_data(token_id, logits[token_id], 0.0)
                for token_id in range(n_vocab)
            ]
        )

        # Score the next k tokens with the big model.
        output = target_model(
            context + speculated_text, max_tokens=1, logprobs=True, echo=True
        )


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

    generate_with_speculator(
        prompt="Q: Name the planets in the solar system? A: ",
        speculator_model=speculator_llm,
        target_model=target_llm,
    )

    # output = llm(
    #     "Q: Name the planets in the solar system? A: ",
    #     max_tokens=32,
    #     stop=["Q:", "\n"], # stop generating just before the model would generate a new question
    #     echo=True, # echo the prompt back in the output
    #     logprobs=True, # return logits for all tokens in the prompt + output
    # )
