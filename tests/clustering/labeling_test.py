"""Tests for content-derived topic labelling."""
import pandas as pd

from src.clustering.labeling import (
    build_stop_words,
    suggest_topic_names,
    top_terms_to_mapping,
    topic_sentiment_hooks,
)


def test_stop_words_include_rating_leakage() -> None:
    sw = set(build_stop_words())
    assert {"star", "stars", "five", "four", "three", "two", "one"} <= sw


def test_names_follow_content_not_cluster_id() -> None:
    # Same clusters, IDs deliberately shuffled -> names must track content.
    terms = {
        0: ["battery", "brand", "long", "price"],
        1: ["echo", "alexa", "music", "home"],
        2: ["tablet", "kid", "price", "good"],
        3: ["kindle", "book", "read", "reader"],
    }
    names = suggest_topic_names(terms)
    assert names[0] == "Batteries"
    assert names[1] == "Echo & Alexa Devices"
    assert names[2] == "Fire Tablets"
    assert names[3] == "Kindle E-Readers & Books"


def test_generic_cluster_flagged_and_disambiguated() -> None:
    terms = {0: ["great", "good", "love", "product", "price"],
             1: ["great", "good", "nice", "well", "use"]}
    names = suggest_topic_names(terms)
    assert all(n.startswith("General / uncategorized") for n in names.values())
    assert names[0] != names[1]  # duplicates disambiguated by top term


def test_top_terms_to_mapping_handles_list_column() -> None:
    df = pd.DataFrame({"top_terms": [["tablet", "kid"], ["battery", "long"]]},
                      index=pd.Index([0, 1], name="cluster_id"))
    mapping = top_terms_to_mapping(df)
    assert mapping == {0: ["tablet", "kid"], 1: ["battery", "long"]}


def test_hooks_rank_by_complaint_rate() -> None:
    clustered = pd.DataFrame({
        "topic_name": ["Batteries"] * 10 + ["Fire Tablets"] * 10,
        "predicted_sentiment": (["negative"] * 4 + ["positive"] * 6
                                + ["negative"] * 1 + ["positive"] * 9),
    })
    hooks = topic_sentiment_hooks(clustered)
    assert hooks.iloc[0]["topic"] == "Batteries"          # worst first
    assert "complaint" in hooks.iloc[0]["hook"].lower()
