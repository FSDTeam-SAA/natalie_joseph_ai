from __future__ import annotations

import json

from app.scripts.seed_companions import CONFIG_DIR, EXPECTED_SLUGS, _build_config_blocks


def _raw_configs() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CONFIG_DIR.iterdir())
        if path.suffix.lower() == ".json"
    ]


def test_exactly_five_named_companion_configs_exist() -> None:
    configs = _raw_configs()
    assert {config["id"] for config in configs} == EXPECTED_SLUGS
    assert {config["name"] for config in configs} == {
        "Elena",
        "Chloé",
        "Thalia",
        "Lina",
        "Luna",
    }


def test_every_companion_has_deployment_voice_and_reference_slots() -> None:
    for config in _raw_configs():
        assert "voice_id" in config["voice"]
        assert isinstance(config["voice"].get("settings"), dict)
        assert isinstance(config["visual_profile"].get("reference_images"), list)
        assert config["visual_profile"].get("generation_instructions")


def test_deployment_asset_maps_override_nonsecret_config() -> None:
    raw = _raw_configs()[0]
    blocks = _build_config_blocks(
        raw,
        voice_id="deployment-voice-id",
        reference_images=["approved/reference.webp"],
    )

    assert blocks["voice_config"]["voice_id"] == "deployment-voice-id"
    assert blocks["visual_config"]["reference_images"] == ["approved/reference.webp"]
