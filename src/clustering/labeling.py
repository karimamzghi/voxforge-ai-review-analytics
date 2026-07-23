"""Post-clustering labelling, validation, and downstream hooks.

This module addresses the five follow-ups from the clustering review. It is split
into two usage points:

FEATURE TIME (before clustering)
    - ``build_stop_words`` — English stop words + domain stop words that kill the
      "five star" rating-word leakage (step 2).

POST CLUSTERING (after ``fit_topic_model``)
    - ``suggest_topic_names`` — derive human-readable names from each cluster's own
      top terms, so labels always follow content even when K-Means re-numbers the
      clusters across runs (step 1). This is the structural fix for the stale-name
      bug: names are recomputed from THIS run, never hardcoded.
    - ``cluster_quality_report`` — a tidy per-cluster QA table (size, top terms,
      sample reviews, a coherence flag) for qualitative validation (step 4).
    - ``topic_sentiment_hooks`` — per-topic sentiment plus a one-line article angle,
      sorted by complaint rate, ready to feed the Task 3 summaries (step 5).

Step 3 (the ~45% generic mass) is handled two ways here: generic clusters are
labelled "General / uncategorized" so they are explicit rather than mislabelled,
and ``cluster_quality_report`` flags them as low-coherence so you can decide
whether to raise ``n_clusters`` or exclude them from the summaries.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.config import DOMAIN_STOP_WORDS

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE TIME — stop words (step 2)
# ─────────────────────────────────────────────────────────────────────────────

# Rating-word leakage: review titles like "five star", "four star" collapse into
# near-duplicate fragments ("five star excelente") that dominate a whole cluster.
# Dropping these tokens forces the model to cluster on the actual product content.

# Generic praise/filler words. NOT stop-worded (they are legitimate sentiment
# content), but used to detect low-information "catch-all" clusters.
GENERIC_TERMS: frozenset[str] = frozenset({
    "great", "good", "love", "loved", "nice", "well", "best", "excellent",
    "product", "products", "price", "buy", "bought", "use", "used", "using",
    "work", "works", "worked", "easy", "get", "got", "really", "would", "like",
})

def build_stop_words(extra: Sequence[str] | None = None) -> list[str]:
    """Return English + domain stop words for the ``TfidfVectorizer``.

    Pass the result as ``TfidfVectorizer(stop_words=build_stop_words())``. If your
    ``fit_topic_model`` builds the vectorizer internally, add a ``stop_words``
    parameter to it and forward this list.
    """
    words = set(ENGLISH_STOP_WORDS) | set(DOMAIN_STOP_WORDS)
    if extra:
        words |= {w.lower() for w in extra}
    return sorted(words)


def drop_low_content_reviews(
    reviews: "pd.DataFrame | pd.Series",
    text_column: str = "classical_text",
    *,
    min_tokens: int = 3,
    stop_words: Sequence[str] | None = None,
    verbose: bool = True,
) -> "pd.DataFrame | pd.Series":
    """Drop reviews with too few meaningful tokens (after stop words).

    Rating-only fragments like "five star ordered" have almost no content once
    stop words are removed, so they collapse into one near-empty catch-all cluster.
    Removing them before clustering shrinks that catch-all and sharpens the
    remaining product categories.

    Accepts a DataFrame (filtered on ``text_column``) or a Series, and returns the
    same type with the index reset. Run it once, right after you build
    ``model_data`` and before clustering.
    """
    stops = set(stop_words) if stop_words is not None else set(build_stop_words())

    def _n_meaningful(text: object) -> int:
        return sum(1 for tok in str(text).lower().split() if tok and tok not in stops)

    source = reviews if isinstance(reviews, pd.Series) else reviews[text_column]
    keep = source.map(_n_meaningful) >= min_tokens
    kept = reviews[keep].reset_index(drop=True)

    if verbose:
        before, after = len(reviews), len(kept)
        print(f"Kept {after:,} of {before:,} reviews "
              f"({before - after:,} low-content fragments dropped)")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# POST CLUSTERING — content-derived naming (step 1)
# ─────────────────────────────────────────────────────────────────────────────

# Ordered taxonomy: (display name, trigger tokens). Priority is list order, but a
# cluster is assigned the category with the MOST trigger hits among its top terms,
# so ordering only breaks ties. Extend this as new product categories appear.
CATEGORY_TAXONOMY: list[tuple[str, frozenset[str]]] = [
    ("Fire Tablets",              frozenset({"tablet", "tablets", "kids", "kid"})),
    ("Kindle E-Readers & Books",  frozenset({"kindle", "book", "books", "read",
                                             "reading", "reader", "ebook",
                                             "paperwhite", "page", "pages"})),
    ("Fire TV & Streaming",       frozenset({"tv", "stick", "streaming", "remote",
                                             "roku", "channel", "channels"})),
    ("Echo & Alexa Devices",      frozenset({"echo", "alexa", "dot", "speaker",
                                             "music", "smart", "home"})),
    ("Batteries",                 frozenset({"battery", "batteries", "aa", "aaa",
                                             "alkaline", "rayovac", "charge",
                                             "charged", "duracell"})),
    ("Ease of Use",               frozenset({"setup", "install", "installation",
                                             "instructions", "simple", "intuitive"})),
]


def _match_category(top_terms: Sequence[str]) -> tuple[str | None, int]:
    """Return (best category name, hit count) for one cluster's top terms.

    Matching is on whole tokens, so "great tablet" hits the "tablet" trigger while
    "start" never hits "star". Returns (None, 0) when no product category matches.
    """
    tokens_per_term = [set(str(term).lower().split()) for term in top_terms]

    best_name: str | None = None
    best_hits = 0
    best_rank = len(top_terms) + 1

    for name, triggers in CATEGORY_TAXONOMY:
        hit_ranks = [i for i, tokens in enumerate(tokens_per_term) if tokens & triggers]
        if not hit_ranks:
            continue
        hits, first_rank = len(hit_ranks), min(hit_ranks)
        # Prefer more hits; break ties by whichever category appears earliest in
        # the cluster's ranked top-terms list.
        if hits > best_hits or (hits == best_hits and first_rank < best_rank):
            best_name, best_hits, best_rank = name, hits, first_rank

    return best_name, best_hits


def _is_generic(top_terms: Sequence[str], *, threshold: float = 0.6) -> bool:
    """True when the cluster's top terms are mostly generic praise/filler."""
    tokens = [tok for term in top_terms for tok in str(term).lower().split()]
    if not tokens:
        return True
    generic = sum(tok in GENERIC_TERMS for tok in tokens)
    return generic / len(tokens) >= threshold


def suggest_topic_names(
    cluster_terms: Mapping[int, Sequence[str]],
    *,
    top_n: int = 8,
) -> dict[int, str]:
    """Derive a name for each cluster from its own top terms.

    ``cluster_terms`` maps ``cluster_id -> ordered list of top terms``. Build it
    from your ``top_terms`` DataFrame in one line — see ``top_terms_to_mapping``.

    Because names are computed from the current run's content, they stay correct
    even when K-Means assigns different integer IDs on the next run. Duplicate
    names (e.g. two generic clusters) are disambiguated with their top term.
    """
    raw: dict[int, str] = {}
    for cluster_id, terms in cluster_terms.items():
        top = list(terms)[:top_n]
        name, hits = _match_category(top)
        if name is None or (hits == 1 and _is_generic(top)):
            # No product category, or a single weak hit drowned in generic praise.
            name = "General / uncategorized"
        raw[cluster_id] = name

    return _disambiguate(raw, cluster_terms, top_n=top_n)


def _disambiguate(
    names: Mapping[int, str],
    cluster_terms: Mapping[int, Sequence[str]],
    *,
    top_n: int,
) -> dict[int, str]:
    """Append a distinctive top term to any name shared by multiple clusters."""
    counts = Counter(names.values())
    resolved: dict[int, str] = {}
    used_markers: dict[str, set[str]] = {}  # name -> markers already taken
    for cluster_id, name in names.items():
        if counts[name] == 1:
            resolved[cluster_id] = name
            continue
        taken = used_markers.setdefault(name, set())
        top = [str(t) for t in list(cluster_terms[cluster_id])[:top_n]]
        # Prefer a distinctive (non-generic) unused term; then any unused term;
        # finally fall back to the cluster id so the marker is always unique.
        marker = next(
            (t for t in top if not _is_generic([t]) and t not in taken),
            next((t for t in top if t not in taken), f"cluster {cluster_id}"),
        )
        taken.add(marker)
        resolved[cluster_id] = f"{name} ({marker})"
    return resolved


def top_terms_to_mapping(top_terms: pd.DataFrame) -> dict[int, list[str]]:
    """Adapt a ``top_terms`` DataFrame into ``{cluster_id: [terms...]}``.

    Handles both shapes this project has produced:
      * indexed by ``cluster_id`` with a list-valued ``top_terms`` column, and
      * long form with ``cluster_id`` + ``term`` rows.
    """
    if "top_terms" in top_terms.columns:
        return {int(cid): list(terms) for cid, terms in top_terms["top_terms"].items()}
    if {"cluster_id", "term"}.issubset(top_terms.columns):
        return {
            int(cid): grp["term"].tolist()
            for cid, grp in top_terms.groupby("cluster_id")
        }
    raise ValueError(
        "Unrecognised top_terms shape; expected a 'top_terms' column or "
        "'cluster_id' + 'term' columns."
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST CLUSTERING — qualitative validation (step 4)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_quality_report(
    cluster_terms: Mapping[int, Sequence[str]],
    representatives: pd.DataFrame,
    *,
    sizes: "Mapping[int, int] | pd.DataFrame | pd.Series | None" = None,
    names: Mapping[int, str] | None = None,
    top_n: int = 8,
    sample_reviews: int = 3,
    text_column: str = "review_text",
    cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """One tidy QA row per cluster for the write-up.

    Since silhouette is weak for short review text, cluster quality is judged
    qualitatively: coherent top terms + on-topic sample reviews. The ``coherent``
    flag is a heuristic (False = generic catch-all worth raising k or excluding).

    ``sizes`` accepts a mapping, a Series, or a DataFrame such as the one
    ``cluster_size_summary`` returns (a ``cluster_id`` column plus a count column).
    """
    size_map = _to_size_mapping(sizes, cluster_column=cluster_column)
    name_map = dict(names) if names is not None else {}

    rows = []
    for cluster_id, terms in cluster_terms.items():
        top = list(terms)[:top_n]
        samples = representatives.loc[
            representatives[cluster_column] == cluster_id, text_column
        ].head(sample_reviews).tolist()

        rows.append({
            "cluster_id": cluster_id,
            "name": name_map.get(cluster_id, ""),
            "size": size_map.get(cluster_id, pd.NA),
            "top_terms": ", ".join(map(str, top)),
            "sample_reviews": " | ".join(samples),
            "coherent": not _is_generic(top),
        })

    report = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    if size_map:
        total = sum(size_map.values())
        report["pct"] = (report["size"] / total * 100).round(2)
    return report


def _to_size_mapping(sizes, *, cluster_column: str = "cluster_id") -> dict[int, int]:
    """Coerce a mapping / Series / DataFrame of cluster sizes into ``{id: count}``."""
    if sizes is None:
        return {}
    if isinstance(sizes, pd.DataFrame):
        id_col = cluster_column if cluster_column in sizes.columns else sizes.columns[0]
        count_col = next(
            (c for c in ("review_count", "size", "count", "n_reviews", "n")
             if c in sizes.columns),
            sizes.columns[-1],  # fall back to the last column
        )
        return {int(cid): int(cnt)
                for cid, cnt in zip(sizes[id_col], sizes[count_col])}
    if isinstance(sizes, pd.Series):
        return {int(cid): int(cnt) for cid, cnt in sizes.items()}
    return {int(cid): int(cnt) for cid, cnt in dict(sizes).items()}


# ─────────────────────────────────────────────────────────────────────────────
# POST CLUSTERING — Task 3 hook (step 5)
# ─────────────────────────────────────────────────────────────────────────────

def topic_sentiment_hooks(
    clustered: pd.DataFrame,
    *,
    topic_column: str = "topic_name",
    sentiment_column: str = "predicted_sentiment",
) -> pd.DataFrame:
    """Per-topic sentiment plus a one-line article angle, sorted by complaint rate.

    Output columns: topic, reviews, positive_pct, neutral_pct, negative_pct, hook.
    Feed straight into the Task 3 GenAI summaries — the topic with the highest
    negative_pct is the one whose article should lead with reliability concerns.
    """
    counts = (
        clustered.groupby([topic_column, sentiment_column]).size()
        .unstack(fill_value=0)
    )
    for label in ("positive", "neutral", "negative"):
        if label not in counts.columns:
            counts[label] = 0

    totals = counts[["positive", "neutral", "negative"]].sum(axis=1)
    summary = pd.DataFrame({
        "topic": counts.index,
        "reviews": totals.values,
        "positive_pct": (counts["positive"] / totals * 100).round(2).values,
        "neutral_pct": (counts["neutral"] / totals * 100).round(2).values,
        "negative_pct": (counts["negative"] / totals * 100).round(2).values,
    }).sort_values("negative_pct", ascending=False).reset_index(drop=True)

    worst = summary["negative_pct"].max()

    def _hook(row: pd.Series) -> str:
        if row["negative_pct"] == worst and worst > 0:
            return (f"Highest complaint rate ({row['negative_pct']:.0f}% negative) — "
                    f"lead the article with the recurring problems.")
        if row["positive_pct"] >= 95:
            return (f"Strongly liked ({row['positive_pct']:.0f}% positive) — "
                    f"frame as a confident recommendation.")
        return (f"Mixed reception ({row['positive_pct']:.0f}% positive / "
                f"{row['negative_pct']:.0f}% negative) — contrast best vs worst picks.")

    summary["hook"] = summary.apply(_hook, axis=1)
    return summary
