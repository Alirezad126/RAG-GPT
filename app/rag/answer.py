import logging

from app.config import DEFAULT_MODEL, GROQ_API_KEY, MAX_TOKENS, MODELS
from app.rag.retrieval import top_chunks

logger = logging.getLogger(__name__)

CONTEXTUALIZE_PROMPT = (
    "Given the chat history and the latest user question, which might reference "
    "earlier turns, rewrite it as a standalone question that can be understood "
    "on its own. Do NOT answer it. Return it unchanged if it already stands "
    "alone. Reply with the question only."
)

ANSWER_PROMPT = (
    "You answer questions about the document excerpts below. Decline anything "
    "unrelated to the document. Keep answers to three sentences at most, and say "
    "you don't know rather than guessing.\n\n"
    "Document excerpts:\n{context}"
)

_client = None
_available = None


def client():
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def available_models():
    """The allowlisted models the provider is actually serving right now.

    Groq retires models from its free catalogue without much notice, so the
    list is intersected with what the API reports rather than trusted blindly.
    If that call fails, fall back to the full allowlist and let the request
    surface the error.
    """
    global _available
    if _available is None:
        try:
            served = {model.id for model in client().models.list().data}
            logger.info("Provider is serving: %s", sorted(served))
            _available = [model for model in MODELS if model in served] or list(MODELS)
        except Exception:  # noqa: BLE001 - provider down; don't cache a failure
            return list(MODELS)
    return _available


def default_model():
    models = available_models()
    return DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]


def resolve(model):
    """The model to answer with: the requested one if served, else the default."""
    return model if model in available_models() else default_model()


def complete(messages, model, max_tokens=MAX_TOKENS):
    response = client().chat.completions.create(
        model=model, messages=messages, temperature=0.3, max_tokens=max_tokens
    )
    return (response.choices[0].message.content or "").strip()


def as_messages(history):
    return [
        {
            "role": "user" if turn.type == "user" else "assistant",
            "content": turn.message,
        }
        for turn in history
    ]


def standalone_question(question, history, model):
    """Fold the conversation into the question so retrieval can stand alone."""
    if not history:
        return question
    messages = (
        [{"role": "system", "content": CONTEXTUALIZE_PROMPT}]
        + as_messages(history)
        + [{"role": "user", "content": question}]
    )
    # A rewrite is one sentence; if the model rambles, fall back to the original.
    rewritten = complete(messages, model, max_tokens=120)
    return rewritten if rewritten else question


def answer(question, history, chunks, vectors, model=None):
    model = resolve(model)
    query = standalone_question(question, history, model)
    context = "\n\n".join(top_chunks(query, chunks, vectors))
    messages = (
        [{"role": "system", "content": ANSWER_PROMPT.format(context=context)}]
        + as_messages(history)
        + [{"role": "user", "content": question}]
    )
    return complete(messages, model)
