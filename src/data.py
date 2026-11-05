import hashlib
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


def make_sample_id(pair):
    identity = [
        pair["url"],
        pair["source_section_index"],
        pair["document"],
        pair["summary"],
    ]
    identity_text = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
    return "vi_" + digest


def clean_wikilingua_pairs(pairs):
    from src.preprocessing import normalize_text

    cleaned_pairs = []
    removed_samples = []
    removed_counts = {}
    source_groups = {}
    representatives = {}
    retained_ids = set()

    for input_index, pair in enumerate(pairs):
        reason = None
        for field in ["document", "summary", "url"]:
            if field not in pair or pair[field] is None:
                reason = "missing_" + field
            elif not isinstance(pair[field], str):
                reason = "invalid_type_" + field
            elif pair[field].strip() == "":
                reason = "empty_" + field
            if reason is not None:
                break

        removal = {
            "input_index": input_index,
            "source_row_index": pair.get("source_row_index"),
            "source_section_index": pair.get("source_section_index"),
            "url": pair.get("url"),
        }

        if reason is None:
            section_index = pair.get("source_section_index")
            if type(section_index) is not int or section_index < 0:
                raise ValueError(f"Invalid source section index at input {input_index}")
            for field in ["document", "summary"]:
                if normalize_text(pair[field]) != pair[field]:
                    raise ValueError(f"Normalize {field} before cleaning input {input_index}")

            document = pair["document"]
            summary = pair["summary"]
            url = pair["url"]
            if document not in source_groups:
                source_groups[document] = set()
            source_groups[document].add(url)

            sample_id = make_sample_id(pair)
            content_key = (document, summary)
            if content_key in representatives:
                reason = "exact_duplicate_pair"
                removal["id"] = sample_id
                removal["representative_id"] = representatives[content_key]
            else:
                if sample_id in retained_ids:
                    raise ValueError(f"Sample ID collision at input {input_index}")
                cleaned_pair = pair.copy()
                cleaned_pair["id"] = sample_id
                cleaned_pair["group_id"] = url
                cleaned_pairs.append(cleaned_pair)
                retained_ids.add(sample_id)
                representatives[content_key] = sample_id

        if reason is not None:
            removal["reason"] = reason
            removed_samples.append(removal)
            if reason not in removed_counts:
                removed_counts[reason] = 0
            removed_counts[reason] = removed_counts[reason] + 1

    shared_source_groups = []
    for document in sorted(source_groups):
        urls = source_groups[document]
        if len(urls) > 1:
            shared_source_groups.append({
                "document_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
                "group_ids": sorted(urls),
            })

    report = {
        "input_pair_count": len(pairs),
        "retained_pair_count": len(cleaned_pairs),
        "removed_pair_count": len(removed_samples),
        "removed_counts_by_reason": removed_counts,
        "removed_samples": removed_samples,
        "shared_source_groups": shared_source_groups,
        "rules": {
            "validation_order": ["document", "summary", "url"],
            "removal_reason": "first failing field only; exact duplicates checked last",
            "sample_id": "vi_ + SHA-256 of UTF-8 compact JSON [url, source_section_index, normalized document, normalized summary]",
            "group_id": "original article URL",
            "duplicate_comparison": "exact document and summary string equality",
            "representative": "first valid pair in pinned source order",
            "source_groups": "recorded from all valid pairs before exact pair removal; overlapping groups must stay together in G2.5",
        },
    }
    return cleaned_pairs, report


def save_cleaning_manifest(report, manifest_path, config_path="configs/train.json"):
    config = load_train_config(config_path)
    manifest = report.copy()
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["dataset"] = config["dataset"].copy()
    output_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
