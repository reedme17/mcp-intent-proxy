"""Non-LLM baselines for intent classification, for comparison in §5.2.

Three baselines, each deliberately "dumb" to show what the LLM classifier
buys over cheap alternatives:

- keyword: grep action verbs in the tool name + description.
- oauth_scope: map the tool to a coarse read/write/admin scope, then to the
  action categories that scope would permit — models OAuth-style coarse grants.
- embedding_nn: nearest-neighbor over tool name+description embeddings, using
  leave-one-out so a tool never retrieves its own label.

All return the same shape as the LLM classifier's action list so the eval
harness can score them with identical criteria. Baselines predict ACTION only
(the dimension with the clearest cheap-heuristic story); sensitivity/externality
are left to the LLM comparison, since keyword/scope have no meaningful signal
for them.
"""

from __future__ import annotations

import re

# --- Keyword baseline ---------------------------------------------------

# Ordered action -> trigger words. Matched against lowercased name+description.
_KEYWORDS: list[tuple[str, list[str]]] = [
    ("DELETE", ["delete", "remove", "destroy", "drop", "clear", "purge", "revoke", "cancel"]),
    ("SEND", ["send", "email", "sms", "message", "post", "publish", "notify", "call", "dispatch", "tweet"]),
    ("SPEND", ["pay", "purchase", "buy", "charge", "checkout", "order", "book", "transaction", "transfer", "invoice"]),
    ("EXECUTE", ["execute", "run", "deploy", "eval", "exec", "trigger", "invoke", "command", "script"]),
    ("CREDENTIAL/IDENTITY", ["password", "token", "secret", "credential", "api_key", "apikey", "auth", "oauth", "login"]),
    ("PHYSICAL", ["device", "light", "thermostat", "lock", "smart home", "actuate", "turn on", "turn off"]),
    ("CREATE", ["create", "add", "new", "insert", "write", "upload", "make", "generate", "register"]),
    ("MODIFY/MANAGE", ["update", "modify", "edit", "change", "set", "manage", "configure", "provision", "move", "rename", "patch"]),
    ("READ/SEARCH", ["get", "read", "list", "search", "find", "fetch", "retrieve", "query", "view", "show", "lookup", "describe"]),
]


def keyword_classify(name: str, description: str) -> list[str]:
    text = f"{name} {description}".lower()
    hits = [action for action, words in _KEYWORDS if any(w in text for w in words)]
    return hits or ["OTHER"]


# --- OAuth-scope baseline -----------------------------------------------

# Coarse scope inferred from verbs, then expanded to the actions that scope
# would grant. This is intentionally coarse — the point is to show that a
# read/write/admin grant cannot separate create/delete/send/spend.
_ADMIN_WORDS = ["delete", "remove", "drop", "manage", "configure", "provision", "admin", "revoke", "deploy", "execute"]
_WRITE_WORDS = ["create", "add", "update", "modify", "edit", "write", "set", "insert", "send", "post", "upload", "pay", "book", "order"]

_SCOPE_TO_ACTIONS = {
    # A read grant permits only reading.
    "read": ["READ/SEARCH"],
    # A write grant lumps every mutation together — the coarseness we expose.
    "write": ["CREATE", "MODIFY/MANAGE", "SEND", "SPEND"],
    # An admin grant additionally permits destruction/execution.
    "admin": ["CREATE", "MODIFY/MANAGE", "SEND", "SPEND", "DELETE", "EXECUTE", "CREDENTIAL/IDENTITY"],
}


def oauth_scope_classify(name: str, description: str) -> list[str]:
    text = f"{name} {description}".lower()
    if any(w in text for w in _ADMIN_WORDS):
        scope = "admin"
    elif any(w in text for w in _WRITE_WORDS):
        scope = "write"
    else:
        scope = "read"
    return list(_SCOPE_TO_ACTIONS[scope])


# --- Embedding nearest-neighbor baseline --------------------------------

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embedding_nn_classify(
    texts: list[str], gt_actions: list[list[str]]
) -> list[list[str]]:
    """Leave-one-out nearest-neighbor over the corpus.

    For each item, embed its name+description, find the most similar OTHER
    item, and copy that neighbor's ground-truth action labels. Leave-one-out
    prevents a tool from retrieving its own label.
    """
    import numpy as np

    model = _get_model()
    emb = model.encode(texts, normalize_embeddings=True)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)  # exclude self
    preds = []
    for i in range(len(texts)):
        j = int(sims[i].argmax())
        preds.append(list(gt_actions[j]))
    return preds
