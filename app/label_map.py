from __future__ import annotations

DOG_HINTS = {
    "terrier",
    "retriever",
    "spaniel",
    "hound",
    "poodle",
    "collie",
    "shepherd",
    "pug",
    "bulldog",
    "beagle",
    "dalmatian",
    "chihuahua",
    "maltese",
    "samoyed",
    "husky",
    "boxer",
    "doberman",
    "shiba",
    "akita",
    "rottweiler",
    "dog",
    "canine",
}

CAT_HINTS = {
    "cat",
    "tabby",
    "tiger cat",
    "persian cat",
    "siamese",
    "lynx",
}


def coarse_label(raw_label: str) -> str:
    normalized = raw_label.lower()
    if any(keyword in normalized for keyword in DOG_HINTS):
        return "dog"
    if any(keyword in normalized for keyword in CAT_HINTS):
        return "cat"
    cleaned = raw_label.split(",", maxsplit=1)[0].strip().lower()
    return cleaned.replace(" ", "_")
