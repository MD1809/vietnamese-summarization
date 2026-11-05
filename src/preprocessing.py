import unicodedata


def normalize_text(text):
    if not isinstance(text, str):
        raise TypeError("normalize_text requires a string")

    normalized_text = unicodedata.normalize("NFC", text)
    words = normalized_text.split()
    normalized_text = " ".join(words)
    return normalized_text


def normalize_pairs(pairs):
    normalized_pairs = []

    for pair in pairs:
        normalized_pair = pair.copy()

        for field in ["document", "summary"]:
            value = pair.get(field)
            if isinstance(value, str):
                normalized_pair[field] = normalize_text(value)

        normalized_pairs.append(normalized_pair)

    return normalized_pairs
