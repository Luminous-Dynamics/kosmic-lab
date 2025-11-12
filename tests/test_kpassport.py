from __future__ import annotations

from pathlib import Path

import jsonschema
import pytest

from core.config import ConfigBundle
from core.kpass import KPassportError, KPassportWriter


def _schema_path() -> Path:
    return Path("schemas/k_passport.json")


@pytest.mark.skipif(not _schema_path().exists(), reason="Schema missing in test env.")
def test_build_and_write_passport(tmp_path: Path) -> None:
    writer = KPassportWriter(_schema_path())
    config = ConfigBundle(payload={"foo": "bar"}, sha256="abc123")
    record = writer.build_record(
        experiment="fre_phase1",
        params={"energy": 0.5},
        estimators={"phi": "phi_E", "te": {"estimator": "ksg", "k": 5, "lag": 1}},
        metrics={"K": 1.1},
        config=config,
        seed=42,
    )
    path = writer.write(record, tmp_path)
    assert path.exists()


def test_validation_failure(tmp_path: Path) -> None:
    schema_path = _schema_path()
    if not schema_path.exists():
        pytest.skip("Schema missing in test env.")

    writer = KPassportWriter(schema_path)
    bad_record = {
        "run_id": "not-a-uuid",
        "commit": "",
        "config_hash": "",
        "seed": 0,
        "experiment": "test",
        "params": {},
        "estimators": {"phi": "phi_E", "te": {"estimator": "ksg", "k": 5, "lag": 1}},
        "metrics": {},
        "timestamp": "invalid",
    }
    with pytest.raises(KPassportError):
        writer.write(bad_record, tmp_path)
