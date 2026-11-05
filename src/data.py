import hashlib
import json
import random
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


def _build_split_groups(pairs, shared_source_groups):
    neighbors = {}
    active_urls = set()
    for pair in pairs:
        url = pair["group_id"]
        active_urls.add(url)
        neighbors[url] = set()

    for source_group in shared_source_groups:
        urls = source_group["group_ids"]
        for url in urls:
            if url not in neighbors:
                neighbors[url] = set()
        if len(urls) > 0:
            first_url = urls[0]
            for url in urls[1:]:
                neighbors[first_url].add(url)
                neighbors[url].add(first_url)

    visited = set()
    groups = {}
    for url in sorted(neighbors):
        if url in visited:
            continue
        pending = [url]
        visited.add(url)
        component = []
        has_active_url = False
        while len(pending) > 0:
            current = pending.pop()
            component.append(current)
            if current in active_urls:
                has_active_url = True
            for neighbor in sorted(neighbors[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        if has_active_url:
            component.sort()
            groups[component[0]] = component
    return groups


def validate_dataset_splits(splits, original_pairs, shared_source_groups):
    split_names = ["train", "validation", "test"]
    if set(splits) != set(split_names):
        raise ValueError("Expected train, validation and test splits")

    expected_ids = set()
    for pair in original_pairs:
        if pair["id"] in expected_ids:
            raise ValueError("Duplicate input sample ID")
        expected_ids.add(pair["id"])

    seen_ids = set()
    url_owners = {}
    document_owners = {}
    for split_name in split_names:
        if len(splits[split_name]) == 0:
            raise ValueError(f"Empty split: {split_name}")
        for pair in splits[split_name]:
            sample_id = pair["id"]
            url = pair["group_id"]
            document = pair["document"]
            if sample_id in seen_ids:
                raise ValueError("Sample ID appears more than once in splits")
            seen_ids.add(sample_id)
            if url != pair["url"]:
                raise ValueError("group_id does not match original URL")
            if url in url_owners and url_owners[url] != split_name:
                raise ValueError("Article group overlaps across splits")
            if document in document_owners and document_owners[document] != split_name:
                raise ValueError("Normalized document overlaps across splits")
            url_owners[url] = split_name
            document_owners[document] = split_name

    if seen_ids != expected_ids:
        raise ValueError("Split sample IDs do not match input sample IDs")

    linked_groups = _build_split_groups(original_pairs, shared_source_groups)
    for group_id in linked_groups:
        owners = set()
        for url in linked_groups[group_id]:
            if url in url_owners:
                owners.add(url_owners[url])
        if len(owners) > 1:
            raise ValueError("Linked article groups are assigned to different splits")

    return {
        "sample_coverage_check_passed": True,
        "unique_id_check_passed": True,
        "group_overlap_check_passed": True,
        "document_overlap_check_passed": True,
        "linked_group_check_passed": True,
    }


def split_wikilingua_pairs(pairs, cleaning_report, config_path="configs/train.json"):
    config = load_train_config(config_path)
    if cleaning_report["retained_pair_count"] != len(pairs):
        raise ValueError("Cleaning report count does not match input pairs")

    shared_source_groups = cleaning_report["shared_source_groups"]
    groups = _build_split_groups(pairs, shared_source_groups)
    group_ids = sorted(groups)
    generator = random.Random(config["seed"])
    generator.shuffle(group_ids)
    train_end = len(group_ids) * 80 // 100
    validation_end = train_end + len(group_ids) * 10 // 100
    if train_end == 0 or validation_end == train_end or validation_end == len(group_ids):
        raise ValueError("Not enough independent groups for non-empty 80/10/10 splits")

    split_group_ids = {
        "train": group_ids[:train_end],
        "validation": group_ids[train_end:validation_end],
        "test": group_ids[validation_end:],
    }
    url_splits = {}
    splits = {}
    for split_name in split_group_ids:
        splits[split_name] = []
        for group_id in split_group_ids[split_name]:
            for url in groups[group_id]:
                url_splits[url] = split_name

    pairs_by_id = {}
    for pair in pairs:
        if pair["id"] in pairs_by_id:
            raise ValueError("Duplicate input sample ID")
        pairs_by_id[pair["id"]] = pair
    for sample_id in sorted(pairs_by_id):
        pair = pairs_by_id[sample_id]
        split_name = url_splits[pair["group_id"]]
        splits[split_name].append(pair.copy())

    checks = validate_dataset_splits(splits, pairs, shared_source_groups)
    split_metadata = {}
    for split_name in splits:
        sample_ids = []
        urls = set()
        for pair in splits[split_name]:
            sample_ids.append(pair["id"])
            urls.add(pair["group_id"])
        split_metadata[split_name] = {
            "pair_count": len(sample_ids),
            "article_group_count": len(urls),
            "split_group_count": len(split_group_ids[split_name]),
            "sample_ids": sample_ids,
            "group_ids": sorted(urls),
            "split_group_ids": sorted(split_group_ids[split_name]),
        }

    manifest = {
        "dataset": config["dataset"].copy(),
        "seed": config["seed"],
        "input_pair_count": len(pairs),
        "target_group_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "rules": {
            "grouping": "original URL plus connected shared-source URL groups, including removed bridge URLs",
            "ordering": "sort component identifiers, shuffle with random.Random(seed); sort sample IDs within each split",
            "allocation": "floor(80% groups) train, floor(10% groups) validation, remainder test",
            "component_id": "lexicographically smallest original URL in connected component",
        },
        "split_groups": groups,
        "splits": split_metadata,
        "checks": checks,
    }
    return splits, manifest


def save_split_manifest(manifest, manifest_path):
    output = manifest.copy()
    output["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
