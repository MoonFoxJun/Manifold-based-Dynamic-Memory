"""Feature extraction and the lexical index.

The index is a TF-IDF matrix over *character bigrams* (Chinese) plus *words*
(Latin scripts).  It is deliberately lexical: cheap, deterministic, and
interpretable.  See ``README`` for the limits of this choice.
"""

import collections
import math
import re

import numpy as np

# Filled by ``build_index``: document frequency and corpus size, used by
# ``query_vector`` to drop stop-word-like bigrams (they appear in a large
# fraction of messages and only add noise to the query).
_DF = None
_N = 0


def grams(text):
    """Tokenize text into ``c:`` (CJK bigram) and ``w:`` (Latin word) tokens."""
    toks = []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", text):
        if len(w) >= 2:
            toks.append("w:" + w.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            toks.append("c:" + run)
        for i in range(len(run) - 1):
            toks.append("c:" + run[i:i + 2])
    return collections.Counter(toks)


def build_index(msgs, vocab_size=3000, min_df=None):
    """Build the TF-IDF matrix and the vocabulary.

    Parameters
    ----------
    msgs : list of dict
        Each message needs a ``text`` field.
    vocab_size : int
        Keep at most this many tokens (most frequent first).
    min_df : int or None
        Drop tokens appearing in fewer than ``min_df`` messages.  ``None``
        means "1 for small archives, 2 otherwise", so a freshly written message
        is still retrievable in a small archive.

    Returns
    -------
    (X, idx, idf)
        ``X`` is the L2-normalized ``n x |vocab|`` matrix, ``idx`` maps token
        to column, ``idf`` is the per-column inverse document frequency.
    """
    global _DF, _N
    if min_df is None:
        min_df = 1 if len(msgs) < 50 else 2
    df = collections.Counter()
    docs = []
    for m in msgs:
        g = grams(m["text"])
        docs.append(g)
        for w in g:
            df[w] += 1

    vocab = [w for w, c in df.most_common(vocab_size) if c >= min_df]
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(msgs)

    X = np.zeros((n, len(vocab)))
    for i, g in enumerate(docs):
        for w, c in g.items():
            if w in idx:
                X[i, idx[w]] = (1 + math.log(c)) * math.log(n / df[w])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms

    idf = np.array([math.log(n / df[w]) for w in vocab])
    _DF, _N = df, n
    return X, idx, idf


def query_vector(query, idx, idf, max_df_frac=0.20, top_k=12):
    """Build the query vector: drop stop-word bigrams, keep the top terms.

    A bigram appearing in more than ``max_df_frac`` of the messages behaves
    like a Chinese stop word ("关于", "来了", "这是") and dilutes the query.
    Keeping only the ``top_k`` highest tf-idf terms sharpens it.

    Returns a zero vector when every query term is dropped (the caller then
    reports "no relevant memory").
    """
    g = grams(query)
    cand = []
    # absolute floor: on small corpora "20% of messages" would drop everything
    df_limit = max(5, max_df_frac * _N) if _N else 5
    for w, c in g.items():
        if w in idx:
            if _DF is not None and _DF[w] > df_limit:
                continue
            cand.append(((1 + math.log(c)) * idf[idx[w]], w))
    cand.sort(reverse=True)

    v = np.zeros(len(idx))
    for wt, w in cand[:top_k]:
        v[idx[w]] = wt
    nrm = np.linalg.norm(v)
    return v / nrm if nrm else v
