"""Pytest coverage for LocalVectorDB + vector_math."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_vector_db import LocalVectorDB
from vector_math import VectorDocument, cosine_similarity, euclidean_distance


def test_vector_math_basics() -> None:
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) == pytest.approx(0.0)
    assert euclidean_distance(a, b) == pytest.approx(0.0)
    assert euclidean_distance(a, c) == pytest.approx(2.0 ** 0.5)


def test_add_documents() -> None:
    db = LocalVectorDB()
    db.add_document(VectorDocument(text="alpha", vector=[1.0, 0.0], doc_id="a"))
    db.add_document({"text": "beta", "vector": [0.0, 1.0], "doc_id": "b"})
    assert len(db.documents) == 2
    assert db.documents[0].text == "alpha"
    assert db.documents[1].doc_id == "b"


def test_save_and_load_json_tmp_dir(tmp_path: Path) -> None:
    db = LocalVectorDB()
    db.add_document(VectorDocument(text="one", vector=[0.1, 0.2, 0.3], doc_id="1"))
    db.add_document(VectorDocument(text="two", vector=[0.4, 0.5, 0.6], doc_id="2"))
    path = tmp_path / "state.json"
    db.save_to_disk(path)
    assert path.is_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "documents" in raw
    assert len(raw["documents"]) == 2

    loaded = LocalVectorDB()
    loaded.load_from_disk(path)
    assert len(loaded.documents) == 2
    assert loaded.documents[0].text == "one"
    assert loaded.documents[1].vector == [0.4, 0.5, 0.6]


def test_search_top_k_cosine() -> None:
    db = LocalVectorDB()
    db.add_document(VectorDocument(text="east", vector=[1.0, 0.0], doc_id="east"))
    db.add_document(VectorDocument(text="north", vector=[0.0, 1.0], doc_id="north"))
    db.add_document(VectorDocument(text="ne", vector=[0.7, 0.7], doc_id="ne"))

    hits = db.search_top_k([1.0, 0.0], k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "east"
    assert hits[1].doc_id == "ne"

    single = db.search_top_k([0.0, 1.0], k=1)
    assert len(single) == 1
    assert single[0].doc_id == "north"
