import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils import load_train_config


def load_wikilingua(cache_dir, config_path="configs/train.json"):
    from datasets import load_dataset

    config = load_train_config(config_path)
    source = config["dataset"]
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        source["id"],
        name=source["language"],
        revision=source["revision"],
        cache_dir=str(cache_path),
    )
    return dataset


def save_source_manifest(
    dataset,
    manifest_path,
    cache_dir,
    config_path="configs/train.json",
):
    from huggingface_hub import DatasetCard, hf_hub_download

    config = load_train_config(config_path)
    source = config["dataset"]
    card_path = hf_hub_download(
        repo_id=source["id"],
        filename="README.md",
        repo_type="dataset",
        revision=source["revision"],
        cache_dir=str(Path(cache_dir)),
    )
    card = DatasetCard.load(card_path)
    card_text = Path(card_path).read_text(encoding="utf-8")
    licensing_text = card_text.split("### Licensing Information", 1)[1]
    licensing_text = licensing_text.split("### Citation Information", 1)[0].strip()
    citation_text = card_text.split("### Citation Information", 1)[1]
    citation_text = citation_text.split("### Contributions", 1)[0].strip()

    splits = {}
    for split_name in dataset:
        split = dataset[split_name]
        splits[split_name] = {
            "article_row_count": len(split),
            "features": split.features.to_dict(),
            "fingerprint": split._fingerprint,
        }

    card_url = (
        "https://huggingface.co/datasets/"
        + source["id"]
        + "/blob/"
        + source["revision"]
        + "/README.md"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": source["id"],
        "revision": source["revision"],
        "language": source["language"],
        "row_unit": "article; section pairs have not been extracted",
        "splits": splits,
        "source_card_url": card_url,
        "upstream_url": "https://github.com/esdurmus/Wikilingua",
        "license_metadata": card.data.to_dict().get("license"),
        "licensing_information": licensing_text,
        "license_note": (
            "Card metadata and licensing text name different licenses. "
            "Both are preserved from the pinned source card."
        ),
        "citation_information": citation_text,
    }

    output_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def convert_wikilingua_sections(rows):
    pairs = []
    fields = ["section_name", "document", "summary"]

    for row_index, row in enumerate(rows):
        url = row.get("url")
        article = row.get("article")
        location = f"row={row_index}, url={url}"

        if not isinstance(article, dict):
            raise ValueError(f"Expected parallel article lists: {location}")

        for field in fields:
            if not isinstance(article.get(field), list):
                raise ValueError(f"Expected article.{field} to be a list: {location}")

        section_count = len(article["document"])
        for field in fields:
            if len(article[field]) != section_count:
                raise ValueError(f"Article list lengths do not match: {location}")

        for section_index in range(section_count):
            pair = {
                "source_row_index": row_index,
                "source_section_index": section_index,
                "url": url,
                "section_name": article["section_name"][section_index],
                "document": article["document"][section_index],
                "summary": article["summary"][section_index],
            }
            pairs.append(pair)

    return pairs
