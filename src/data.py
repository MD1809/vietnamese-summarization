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


def _split_manifest_fingerprint(split_manifest):
    metadata = split_manifest.copy()
    metadata.pop("created_at_utc", None)
    text = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_parent_splits(dataset_splits, split_manifest):
    split_names = ["train", "validation", "test"]
    if set(dataset_splits) != set(split_names):
        raise ValueError("Expected train, validation and test splits")

    url_groups = {}
    linked_groups = []
    for group_id in split_manifest["split_groups"]:
        urls = split_manifest["split_groups"][group_id]
        linked_groups.append({"group_ids": urls})
        for url in urls:
            if url in url_groups:
                raise ValueError("URL appears in multiple parent split groups")
            url_groups[url] = group_id

    pairs_by_id = {}
    pairs_by_group = {}
    original_pairs = []
    for split_name in split_names:
        pairs_by_id[split_name] = {}
        pairs_by_group[split_name] = {}
        for pair in dataset_splits[split_name]:
            sample_id = pair["id"]
            if sample_id in pairs_by_id[split_name]:
                raise ValueError("Duplicate sample ID in parent split")
            pairs_by_id[split_name][sample_id] = pair
            original_pairs.append(pair)
            url = pair["group_id"]
            if url not in url_groups:
                raise ValueError("Sample URL missing from parent split groups")
            group_id = url_groups[url]
            if group_id not in pairs_by_group[split_name]:
                pairs_by_group[split_name][group_id] = []
            pairs_by_group[split_name][group_id].append(pair)

        metadata = split_manifest["splits"][split_name]
        if set(pairs_by_id[split_name]) != set(metadata["sample_ids"]):
            raise ValueError("Input sample IDs do not match parent split manifest")
        if len(dataset_splits[split_name]) != metadata["pair_count"]:
            raise ValueError("Input count does not match parent split manifest")
        if set(pairs_by_group[split_name]) != set(metadata["split_group_ids"]):
            raise ValueError("Input groups do not match parent split manifest")

    validate_dataset_splits(dataset_splits, original_pairs, linked_groups)
    return pairs_by_id, pairs_by_group


def select_fixed_subsets(
    dataset_splits,
    split_manifest,
    target_counts,
    config_path="configs/train.json",
):
    config = load_train_config(config_path)
    if split_manifest["dataset"] != config["dataset"]:
        raise ValueError("Parent dataset does not match training configuration")
    if split_manifest["seed"] != config["seed"]:
        raise ValueError("Parent seed does not match training configuration")
    if set(target_counts) != {"train", "validation", "test"}:
        raise ValueError("Expected target counts for train, validation and test")

    pairs_by_id, pairs_by_group = _index_parent_splits(dataset_splits, split_manifest)
    subsets = {}
    subset_metadata = {}
    for split_name in ["train", "validation", "test"]:
        target = target_counts[split_name]
        if type(target) is not int or target <= 0:
            raise ValueError("Subset target must be a positive integer")
        if target > len(pairs_by_id[split_name]):
            raise ValueError(f"Subset target exceeds available pairs in {split_name}")

        group_ids = sorted(pairs_by_group[split_name])
        generator = random.Random(config["seed"])
        generator.shuffle(group_ids)
        selected_group_ids = []
        selected_ids = []
        for group_id in group_ids:
            selected_group_ids.append(group_id)
            for pair in pairs_by_group[split_name][group_id]:
                selected_ids.append(pair["id"])
            if len(selected_ids) >= target:
                break

        selected_ids.sort()
        subsets[split_name] = []
        selected_urls = set()
        for sample_id in selected_ids:
            pair = pairs_by_id[split_name][sample_id]
            subsets[split_name].append(pair.copy())
            selected_urls.add(pair["group_id"])
        subset_metadata[split_name] = {
            "target_pair_count": target,
            "pair_count": len(selected_ids),
            "parent_pair_count": len(dataset_splits[split_name]),
            "article_group_count": len(selected_urls),
            "split_group_count": len(selected_group_ids),
            "sample_ids": selected_ids,
            "group_ids": sorted(selected_urls),
            "split_group_ids": sorted(selected_group_ids),
        }

    manifest = {
        "dataset": config["dataset"].copy(),
        "seed": config["seed"],
        "parent_split_manifest": "wikilingua_splits.json",
        "parent_split_sha256": _split_manifest_fingerprint(split_manifest),
        "rules": {
            "selection": "within each parent split, sort component IDs, shuffle with a fresh random.Random(seed), take whole components until reaching or exceeding target",
            "ordering": "sort selected sample IDs within each subset",
            "reuse": "restore saved sample IDs; do not resample for each training run",
        },
        "splits": subset_metadata,
    }
    restore_fixed_subsets(dataset_splits, split_manifest, manifest)
    manifest["checks"] = {
        "parent_membership_check_passed": True,
        "whole_group_check_passed": True,
        "unique_id_check_passed": True,
        "group_overlap_check_passed": True,
        "document_overlap_check_passed": True,
        "linked_group_check_passed": True,
    }
    return subsets, manifest


def restore_fixed_subsets(dataset_splits, split_manifest, subset_manifest):
    if subset_manifest["parent_split_sha256"] != _split_manifest_fingerprint(split_manifest):
        raise ValueError("Subset manifest refers to a different parent split manifest")
    if subset_manifest["dataset"] != split_manifest["dataset"]:
        raise ValueError("Subset and parent datasets do not match")
    if subset_manifest["seed"] != split_manifest["seed"]:
        raise ValueError("Subset and parent seeds do not match")
    if set(subset_manifest["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Expected three saved subsets")

    pairs_by_id, pairs_by_group = _index_parent_splits(dataset_splits, split_manifest)
    restored = {}
    selected_pairs = []
    for split_name in ["train", "validation", "test"]:
        metadata = subset_manifest["splits"][split_name]
        sample_ids = metadata["sample_ids"]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Duplicate saved subset sample ID")
        expected_ids = set()
        for group_id in metadata["split_group_ids"]:
            if group_id not in pairs_by_group[split_name]:
                raise ValueError("Saved group does not belong to original split")
            for pair in pairs_by_group[split_name][group_id]:
                expected_ids.add(pair["id"])
        if set(sample_ids) != expected_ids:
            raise ValueError("Saved subset is not a collection of whole parent groups")
        if len(sample_ids) != metadata["pair_count"]:
            raise ValueError("Saved subset count does not match sample IDs")

        restored[split_name] = []
        urls = set()
        for sample_id in sorted(sample_ids):
            pair = pairs_by_id[split_name][sample_id].copy()
            restored[split_name].append(pair)
            selected_pairs.append(pair)
            urls.add(pair["group_id"])
        if set(metadata["group_ids"]) != urls:
            raise ValueError("Saved article groups do not match selected samples")
        if metadata["article_group_count"] != len(urls):
            raise ValueError("Saved article group count does not match samples")
        if metadata["split_group_count"] != len(metadata["split_group_ids"]):
            raise ValueError("Saved split group count does not match groups")
        if metadata["parent_pair_count"] != len(dataset_splits[split_name]):
            raise ValueError("Saved parent count does not match original split")
        target = metadata["target_pair_count"]
        if type(target) is not int or target <= 0 or len(sample_ids) < target:
            raise ValueError("Saved subset does not meet a valid target")

    links = []
    for urls in split_manifest["split_groups"].values():
        links.append({"group_ids": urls})
    validate_dataset_splits(restored, selected_pairs, links)
    return restored


def save_subset_manifest(manifest, manifest_path):
    return save_split_manifest(manifest, manifest_path)


def summarize_token_lengths(lengths):
    if len(lengths) == 0:
        raise ValueError("Cannot summarize empty token lengths")
    ordered = sorted(lengths)
    rank = (95 * len(ordered) + 99) // 100
    return {
        "mean": sum(ordered) / len(ordered),
        "p95": ordered[rank - 1],
        "max": ordered[-1],
    }


def _validate_export_subsets(subset_splits, subset_manifest):
    all_pairs = []
    for split_name in ["train", "validation", "test"]:
        metadata = subset_manifest["splits"][split_name]
        sample_ids = []
        urls = set()
        for pair in subset_splits[split_name]:
            sample_ids.append(pair["id"])
            urls.add(pair["group_id"])
            all_pairs.append(pair)
        if sample_ids != sorted(metadata["sample_ids"]):
            raise ValueError("Export IDs/order do not match fixed subset manifest")
        if len(sample_ids) != metadata["pair_count"]:
            raise ValueError("Export count does not match fixed subset manifest")
        if urls != set(metadata["group_ids"]):
            raise ValueError("Export article groups do not match subset manifest")
    validate_dataset_splits(subset_splits, all_pairs, [])


def build_dataset_statistics(
    subset_splits,
    subset_manifest,
    cleaning_report,
    config_path="configs/train.json",
):
    from transformers import AutoTokenizer

    config = load_train_config(config_path)
    if subset_manifest["dataset"] != config["dataset"]:
        raise ValueError("Statistics dataset does not match configuration")
    if subset_manifest["seed"] != config["seed"]:
        raise ValueError("Statistics seed does not match configuration")
    _validate_export_subsets(subset_splits, subset_manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        use_fast=False,
    )
    document_lengths = []
    summary_lengths = []
    for pair in subset_splits["train"]:
        encoded_document = tokenizer(
            text=pair["document"],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        encoded_summary = tokenizer(
            text_target=pair["summary"],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        document_lengths.append(len(encoded_document["input_ids"]))
        summary_lengths.append(len(encoded_summary["input_ids"]))

    split_counts = {}
    for split_name in ["train", "validation", "test"]:
        metadata = subset_manifest["splits"][split_name]
        split_counts[split_name] = {
            "pair_count": metadata["pair_count"],
            "article_count": metadata["article_group_count"],
        }
    statistics = {
        "dataset": config["dataset"].copy(),
        "seed": config["seed"],
        "subset_manifest_sha256": _split_manifest_fingerprint(subset_manifest),
        "splits": split_counts,
        "removed_counts_by_reason": cleaning_report["removed_counts_by_reason"].copy(),
        "training_token_lengths": {
            "sample_count": len(subset_splits["train"]),
            "document": summarize_token_lengths(document_lengths),
            "summary": summarize_token_lengths(summary_lengths),
        },
        "tokenization": {
            "model_id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "tokenizer_class": type(tokenizer).__name__,
            "use_fast": False,
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
            "summary_input": "text_target",
            "p95_method": "nearest rank: sorted_lengths[ceil(0.95 * N) - 1]",
        },
    }
    return statistics


def _prepared_file_path(project_root, relative_path):
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Prepared file path must stay relative to the project root")
    root = Path(project_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Prepared file path escapes the project root")
    return path


def _read_prepared_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as data_file:
        for line in data_file:
            rows.append(json.loads(line))
    return rows


def verify_prepared_datasets(project_root, processed_manifest, subset_manifest):
    if processed_manifest["subset_manifest_sha256"] != _split_manifest_fingerprint(subset_manifest):
        raise ValueError("Processed data refers to a different subset manifest")
    if processed_manifest["dataset"] != subset_manifest["dataset"]:
        raise ValueError("Processed dataset identity does not match subset manifest")
    if processed_manifest["seed"] != subset_manifest["seed"]:
        raise ValueError("Processed seed does not match subset manifest")
    splits = {}
    for split_name in ["train", "validation", "test"]:
        metadata = processed_manifest["files"][split_name]
        path = _prepared_file_path(project_root, metadata["path"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Prepared file changed: {split_name}")
        if len(content) != metadata["byte_count"]:
            raise ValueError(f"Prepared file size changed: {split_name}")
        rows = _read_prepared_jsonl(path)
        if len(rows) != metadata["pair_count"]:
            raise ValueError(f"Prepared row count does not match manifest: {split_name}")
        splits[split_name] = rows
    _validate_export_subsets(splits, subset_manifest)

    stats_metadata = processed_manifest["statistics"]
    stats_path = _prepared_file_path(project_root, stats_metadata["path"])
    if hashlib.sha256(stats_path.read_bytes()).hexdigest() != stats_metadata["sha256"]:
        raise ValueError("Prepared statistics file changed")
    lock = processed_manifest["test_lock"]
    test_metadata = processed_manifest["files"]["test"]
    if lock["locked"] is not True or lock["sha256"] != test_metadata["sha256"]:
        raise ValueError("Test lock does not match prepared test file")
    return {
        "file_hash_check_passed": True,
        "jsonl_readback_check_passed": True,
        "fixed_sample_ids_check_passed": True,
        "test_lock_check_passed": True,
    }


def export_prepared_datasets(
    subset_splits,
    subset_manifest,
    statistics,
    project_root,
    config_path="configs/train.json",
):
    config = load_train_config(config_path)
    _validate_export_subsets(subset_splits, subset_manifest)
    subset_hash = _split_manifest_fingerprint(subset_manifest)
    if statistics["subset_manifest_sha256"] != subset_hash:
        raise ValueError("Statistics refers to a different subset manifest")
    if statistics["dataset"] != config["dataset"] or subset_manifest["dataset"] != config["dataset"]:
        raise ValueError("Prepared dataset does not match configuration")
    if statistics["seed"] != config["seed"] or subset_manifest["seed"] != config["seed"]:
        raise ValueError("Prepared seed does not match configuration")
    if statistics["tokenization"]["model_id"] != config["model"]["id"]:
        raise ValueError("Statistics tokenizer does not match configuration")
    if statistics["tokenization"]["revision"] != config["model"]["revision"]:
        raise ValueError("Statistics tokenizer revision does not match configuration")
    for split_name in ["train", "validation", "test"]:
        metadata = subset_manifest["splits"][split_name]
        expected_counts = {
            "pair_count": metadata["pair_count"],
            "article_count": metadata["article_group_count"],
        }
        if statistics["splits"][split_name] != expected_counts:
            raise ValueError("Statistics counts do not match fixed subset")

    candidate_files = {}
    file_metadata = {}
    for split_name in ["train", "validation", "test"]:
        relative_path = config["paths"][split_name + "_file"]
        path = _prepared_file_path(project_root, relative_path)
        lines = []
        for pair in subset_splits[split_name]:
            lines.append(json.dumps(pair, ensure_ascii=False, separators=(",", ":")))
        content = ("\n".join(lines) + "\n").encode("utf-8")
        if path in candidate_files:
            raise ValueError("Prepared split paths must be different")
        candidate_files[path] = content
        file_metadata[split_name] = {
            "path": relative_path,
            "pair_count": len(subset_splits[split_name]),
            "byte_count": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    stats_relative = "data/manifests/wikilingua_statistics.json"
    stats_path = _prepared_file_path(project_root, stats_relative)
    stats_content = json.dumps(statistics, ensure_ascii=False, indent=2).encode("utf-8")
    if stats_path in candidate_files:
        raise ValueError("Statistics path must not match a split path")
    candidate_files[stats_path] = stats_content
    manifest_path = _prepared_file_path(project_root, "data/manifests/wikilingua_processed.json")
    if manifest_path in candidate_files:
        raise ValueError("Processed manifest path must not match a data path")

    manifest = {
        "dataset": config["dataset"].copy(),
        "seed": config["seed"],
        "subset_manifest": "data/manifests/wikilingua_subsets.json",
        "subset_manifest_sha256": subset_hash,
        "files": file_metadata,
        "statistics": {
            "path": stats_relative,
            "sha256": hashlib.sha256(stats_content).hexdigest(),
        },
        "test_lock": {
            "locked": True,
            "sha256": file_metadata["test"]["sha256"],
            "policy": "fixed IDs and content; do not use test for tuning; evaluate only in stage 6",
        },
    }
    existing_manifest = None
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_prepared_datasets(project_root, existing_manifest, subset_manifest)
        existing_metadata = existing_manifest.copy()
        existing_metadata.pop("created_at_utc", None)
        if existing_metadata != manifest:
            raise ValueError("Prepared manifest is locked; refusing a different export")

    for path in candidate_files:
        if path.exists() and path.read_bytes() != candidate_files[path]:
            raise ValueError(f"Refusing to overwrite different prepared content: {path.name}")
    for path in candidate_files:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(candidate_files[path])

    for split_name in ["train", "validation", "test"]:
        path = _prepared_file_path(project_root, file_metadata[split_name]["path"])
        restored_rows = _read_prepared_jsonl(path)
        if restored_rows != subset_splits[split_name]:
            raise ValueError(f"JSONL readback differs from selected data: {split_name}")
    verify_prepared_datasets(project_root, manifest, subset_manifest)
    if existing_manifest is not None:
        return existing_manifest
    return save_split_manifest(manifest, manifest_path)
