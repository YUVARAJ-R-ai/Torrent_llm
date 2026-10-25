"""Turning a question into the text a chat model expects, and knowing when it is done.

Shared by every front end (the single-chain API and the pool) so they cannot
drift apart on what "ask the model a question" means.
"""

from __future__ import annotations

from typing import Any

#: Steers a small model toward the short, plain answers a live demo wants.
#: Otherwise it drifts into markdown and long explanations.
DEFAULT_SYSTEM_PROMPT = "Answer in one short, plain sentence. No markdown."


class NoChatTemplate(ValueError):
    """The tokenizer has no chat template, so a question cannot be wrapped."""


def render_chat(tokenizer: Any, prompt: str, system_prompt: str | None) -> str:
    """Wrap a question in the model's own chat template.

    A chat model given bare text treats it as something to continue, not a
    question to answer: "Capital of India" comes back as "is located in the
    state of which country?". The template is what tells it a user asked
    something and it is now the assistant's turn.
    """
    template = getattr(tokenizer, "apply_chat_template", None)
    if template is None:
        raise NoChatTemplate("this model's tokenizer has no chat template")
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    # enable_thinking is Qwen3's switch for its reasoning preamble, which would
    # spend the whole token budget before the answer starts. Templates that do
    # not use it ignore it.
    return template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def stop_token_ids(model_id: str, tokenizer: Any) -> set[int]:
    """Every id the model treats as "done", from its generation config.

    The tokenizer's ``eos_token_id`` alone is not enough: Qwen3 ends a chat turn
    with ``<|im_end|>`` but a raw document with ``<|endoftext|>``, and the
    generation config lists both. Read without loading any weights.
    """
    from transformers import GenerationConfig

    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    try:
        eos = GenerationConfig.from_pretrained(model_id).eos_token_id
    except OSError:
        # No generation config published; the tokenizer's id is the best guess.
        return ids
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(eos)
    return ids
