from __future__ import annotations

from app.label_map import coarse_label


def test_coarse_label_maps_dog_breeds_to_dog():
    assert coarse_label("golden retriever") == "dog"


def test_coarse_label_maps_tabby_to_cat():
    assert coarse_label("tabby cat") == "cat"
