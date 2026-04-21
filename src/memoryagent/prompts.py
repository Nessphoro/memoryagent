"""Chat-template rendering for the Qwen RAG pipeline.

Qwen3.5-0.8B is an instruct model with a chat template baked into the
tokenizer. Feeding it raw ``"Document: ... Question: ... Answer:"`` strings
(as the first draft did) bypasses the format the model was trained on and
silently costs us free quality. These helpers route both the query-encoding
path and the per-doc forward path through ``tokenizer.apply_chat_template``.

Two render targets:

  - ``render_query_prompt(tokenizer, question)`` — the prompt for query-side
    embedding. ``encode_query`` takes the hidden state at the last token (the
    assistant opener) and projects it to BGE space. No document yet — that's
    what we're retrieving.
  - ``render_doc_qa_prefix(tokenizer, doc, question)`` — the templated prefix
    for one of the k retrieved docs. The gold-answer tokens are concatenated
    after this string and contribute to the REPLUG mixture loss.

Both helpers return strings; the caller tokenizes with
``add_special_tokens=False`` because the template already emits its own
``<|im_start|>`` / ``<|im_end|>`` markers.
"""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided "
    "context. Keep answers short, direct, and grounded in the context."
)


def render_query_prompt(tokenizer, question: str) -> str:
    """system + user(question) + assistant opener — no document."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def render_doc_qa_prefix(tokenizer, doc: str, question: str) -> str:
    """system + user(context + question) + assistant opener.

    The assistant opener (``<|im_start|>assistant\\n``) is the last thing in the
    string, so the answer tokens are appended without a leading space.
    """
    user_content = f"Context: {doc}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
