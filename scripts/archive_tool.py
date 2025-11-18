#!/usr/bin/env python3
"""
Archive Track G artifacts (checkpoints + JSONL logs) into timestamped bundles
or verify bundles for reproducibility.

Create:
    python scripts/archive_tool.py create \
        --checkpoint logs/track_g/checkpoints/phase_g2_latest.json \
        --log logs/track_g/episodes/phase_g2.jsonl \
        --config fre/configs/track_g_phase_g2.yaml \
        --output archives/g2_bundle.tar.gz

Verify:
    python scripts/archive_tool.py verify --archive archives/g2_bundle.tar.gz

Diff configs inside archive:
    python scripts/archive_tool.py diff --archive archives/g2_bundle.tar.gz

Summaries:
    python scripts/archive_tool.py summary --archive archives/g2_bundle.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from jsonschema import Draft7Validator

try:
    from scripts.config_registry import lookup as registry_lookup  # type: ignore
except Exception:  # pragma: no cover
    registry_lookup = None
SCHEMA = Draft7Validator(json.loads(Path("schemas/archive_metadata.schema.json").read_text(encoding="utf-8")))


def compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def snapshot_config(config_path: Path, bundle_dir: Path) -> Path:
    snapshot = bundle_dir / "config_snapshot.yaml"
    snapshot.write_bytes(config_path.read_bytes())
    (bundle_dir / "config_hash.txt").write_text(compute_hash(config_path), encoding="utf-8")
    return snapshot


def _extract_checkpoint_snapshot(checkpoint: Path, dest: Path) -> Optional[Path]:
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        snapshot = payload.get("metadata", {}).get("config_snapshot")
        if snapshot:
            snapshot_path = dest / "config_snapshot_from_checkpoint.yaml"
            snapshot_path.write_text(snapshot, encoding="utf-8")
            return snapshot_path
    except json.JSONDecodeError:
        pass
    return None


def create_archive(checkpoint: Path, config: Path, log: Optional[Path], output: Optional[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # If output is provided and is in a temp directory, use that directory
    if output and output.parent != Path("."):
        bundle_dir = output.parent / f"track_g_bundle_{timestamp}"
    else:
        bundle_dir = Path("archives") / f"track_g_bundle_{timestamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dest = bundle_dir / checkpoint.name
    checkpoint_dest.write_bytes(checkpoint.read_bytes())

    metadata: Dict[str, Dict[str, Dict[str, str]] | str] = {
        "timestamp": timestamp,
        "bundle": bundle_dir.name,
        "files": {
            "checkpoint": {"name": checkpoint_dest.name, "hash": compute_hash(checkpoint)},
        },
    }

    if log:
        log_dest = bundle_dir / log.name
        log_dest.write_bytes(log.read_bytes())
        metadata["files"]["log"] = {"name": log_dest.name, "hash": compute_hash(log)}  # type: ignore[index]

    snapshot_path = snapshot_config(config, bundle_dir)
    embedded_snapshot = _extract_checkpoint_snapshot(checkpoint, bundle_dir)
    config_hash = compute_hash(config)
    metadata["files"]["config_snapshot"] = {
        "name": snapshot_path.name,
        "hash": config_hash,
    }  # type: ignore[index]
    if embedded_snapshot:
        metadata["files"]["checkpoint_config_snapshot"] = {
            "name": embedded_snapshot.name,
            "hash": compute_hash(embedded_snapshot),
        }  # type: ignore[index]
    if registry_lookup:
        entry = registry_lookup(hash_value=config_hash)
        if entry:
            metadata["config_label"] = entry.get("label")
            metadata["config_notes"] = entry.get("notes")

    metadata_path = bundle_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    output_path = output or (bundle_dir.with_suffix(".tar.gz"))
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)

    print(f"✅ Archive created: {output_path}")
    return output_path


def _hash_tar_member(tar: tarfile.TarFile, member_name: str) -> str:
    member = tar.getmember(member_name)
    fileobj = tar.extractfile(member)
    if not fileobj:
        raise SystemExit(f"❌ Failed to read {member_name} from archive")
    h = hashlib.sha256()
    while chunk := fileobj.read(65536):
        h.update(chunk)
    return h.hexdigest()


def load_metadata(archive_path: Path) -> Dict[str, Any]:
    with tarfile.open(archive_path, "r:gz") as tar:
        metadata_member = next((m for m in tar.getmembers() if m.name.endswith("metadata.json")), None)
        if not metadata_member:
            raise SystemExit("❌ metadata.json not found in archive")
        metadata_data = tar.extractfile(metadata_member).read().decode("utf-8")  # type: ignore[union-attr]
        metadata = json.loads(metadata_data)
        errors = sorted(SCHEMA.iter_errors(metadata), key=lambda e: e.path)
        if errors:
            msgs = "\n".join(f"- {'/'.join(map(str, err.path))}: {err.message}" for err in errors)
            raise SystemExit(f"❌ metadata schema validation failed:\n{msgs}")
        return metadata


def metadata_json_hash(metadata: Dict[str, Any]) -> str:
    canonical = json.dumps(metadata, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def metadata_to_markdown(metadata: Dict[str, Any]) -> str:
    lines = ["| Field | Value |", "|---|---|"]
    lines.append(f"| Bundle | {metadata.get('bundle','')} |")
    lines.append(f"| Timestamp | {metadata.get('timestamp','')} |")
    lines.append(f"| Config Label | {metadata.get('config_label','')} |")
    lines.append(f"| Config Notes | {metadata.get('config_notes','')} |")
    files = metadata.get("files", {})
    for key, info in files.items():
        lines.append(f"| File: {key} | `{info.get('name','')}` (`{info.get('hash','')}`) |")
    lines.append("")
    lines.append(f"`metadata json sha256: {metadata_json_hash(metadata)}`")
    return "\n".join(lines)


def verify_archive(archive_path: Path) -> None:
    metadata = load_metadata(archive_path)
    files = metadata.get("files", {})
    root = metadata.get("bundle")
    if not root:
        raise SystemExit("❌ metadata missing bundle name")
    with tarfile.open(archive_path, "r:gz") as re_tar:
        members = {m.name: m for m in re_tar.getmembers()}
        for key, info in files.items():
            name = info.get("name")
            expected_hash = info.get("hash")
            if not name or not expected_hash:
                raise SystemExit(f"❌ metadata missing name/hash for {key}")
            member_name = f"{root}/{name}"
            if member_name not in members:
                raise SystemExit(f"❌ member {member_name} missing from archive")
            actual_hash = _hash_tar_member(re_tar, member_name)
            if actual_hash != expected_hash:
                raise SystemExit(f"❌ Hash mismatch for {key}: expected {expected_hash}, got {actual_hash}")
    print(f"✅ Archive verification passed: {archive_path}")


def diff_archive_configs(archive_path: Path, compare_config: Optional[Path] = None) -> str:
    metadata = load_metadata(archive_path)
    root = metadata.get("bundle")
    if not root:
        raise SystemExit("❌ metadata missing bundle name")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        def read_member(name: str) -> str | None:
            full = f"{root}/{name}"
            if full not in members:
                return None
            fileobj = tar.extractfile(members[full])
            if not fileobj:
                return None
            return fileobj.read().decode("utf-8")

        config_yaml = read_member("config_snapshot.yaml")
        ckpt_yaml = read_member("config_snapshot_from_checkpoint.yaml")
    if config_yaml is None:
        raise SystemExit("❌ Archive missing config snapshot.")
    target_yaml = ckpt_yaml
    label_from = "config_snapshot.yaml"
    label_to = "config_snapshot_from_checkpoint.yaml"
    if compare_config:
        target_yaml = compare_config.read_text(encoding="utf-8")
        label_to = str(compare_config)
    if target_yaml is None:
        raise SystemExit("❌ No target snapshot available for diff.")
    import difflib
    diff = difflib.unified_diff(
        config_yaml.splitlines(),
        target_yaml.splitlines(),
        fromfile=label_from,
        tofile=label_to,
        lineterm="",
    )
    return "\n".join(diff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive checkpoints + logs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new archive")
    create_parser.add_argument("--checkpoint", type=Path, required=True)
    create_parser.add_argument("--log", type=Path, required=False)
    create_parser.add_argument("--config", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, default=None, help="Output .tar.gz path")

    verify_parser = subparsers.add_parser("verify", help="Verify an existing archive")
    verify_parser.add_argument("--archive", type=Path, required=True)

    summary_parser = subparsers.add_parser("summary", help="Print archive metadata summary")
    summary_parser.add_argument("--archive", type=Path, required=True)
    summary_parser.add_argument("--markdown", action="store_true", help="Emit summary as Markdown table")
    summary_parser.add_argument("--markdown-path", type=Path, help="Write Markdown summary to this path")
    summary_parser.add_argument("--check-markdown", action="store_true", help="Compare Markdown summary with existing file")

    diff_parser = subparsers.add_parser("diff", help="Diff config snapshots inside an archive")
    diff_parser.add_argument("--archive", type=Path, required=True)
    diff_parser.add_argument("--config", type=Path, help="Compare archive snapshot against this config file")

    args = parser.parse_args()
    if args.command == "create":
        create_archive(args.checkpoint, args.config, args.log, args.output)
    elif args.command == "verify":
        verify_archive(args.archive)
    elif args.command == "summary":
        metadata = load_metadata(args.archive)
        if args.markdown or args.markdown_path or args.check_markdown:
            md_text = metadata_to_markdown(metadata)
            if args.check_markdown:
                if not args.markdown_path:
                    raise SystemExit("❌ --check-markdown requires --markdown-path")
                existing = args.markdown_path.read_text(encoding="utf-8")
                if existing.strip() != md_text.strip():
                    raise SystemExit("❌ Markdown summary mismatch.")
                print("✅ Markdown summary matches.")
            if args.markdown_path and not args.check_markdown:
                args.markdown_path.write_text(md_text + "\n", encoding="utf-8")
                print(f"✅ Markdown summary written to {args.markdown_path}")
            if args.markdown:
                print(md_text)
        else:
            print(json.dumps(metadata, indent=2))
    elif args.command == "diff":
        diff_text = diff_archive_configs(args.archive, args.config)
        if diff_text.strip():
            print(diff_text)
        else:
            print("✅ Config snapshots match.")


if __name__ == "__main__":
    main()
