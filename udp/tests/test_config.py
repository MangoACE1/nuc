from __future__ import annotations

import json

import pytest


VALID_CONFIG = {
    "prediction_topic": "/netcatch/dynamics/prediction",
    "multicast_group": "239.255.42.99",
    "multicast_port": 15150,
    "multicast_ttl": 1,
    "multicast_interface_ip": "0.0.0.0",
    "publish_hz": 30.0,
    "prediction_stale_s": 0.15,
    "max_packet_bytes": 1200,
    "frame_id": "mocap_world_enu",
    "object_name": "obj1",
}


def test_load_config_accepts_complete_valid_config(tmp_path):
    """Catches a missing loader or a loader that alters valid deployment values."""
    try:
        from netcatch_udp.config import load_config
    except ModuleNotFoundError:
        pytest.fail("netcatch_udp.config loader is not implemented")

    path = tmp_path / "config.json"
    path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")

    config = load_config(path)

    assert config.prediction_topic == "/netcatch/dynamics/prediction"
    assert config.multicast_group == "239.255.42.99"
    assert config.multicast_port == 15150
    assert config.multicast_ttl == 1
    assert config.multicast_interface_ip == "0.0.0.0"
    assert config.publish_hz == 30.0
    assert config.prediction_stale_s == 0.15
    assert config.max_packet_bytes == 1200
    assert config.frame_id == "mocap_world_enu"
    assert config.object_name == "obj1"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("prediction_topic", "netcatch/dynamics/prediction"),
        ("multicast_group", "192.168.10.10"),
        ("multicast_group", "ff02::1"),
        ("multicast_port", True),
        ("multicast_port", 0),
        ("multicast_port", 65_536),
        ("multicast_ttl", False),
        ("multicast_ttl", 0),
        ("multicast_ttl", 256),
        ("multicast_interface_ip", "not-an-ip"),
        ("publish_hz", True),
        ("publish_hz", 0.0),
        ("publish_hz", float("inf")),
        ("prediction_stale_s", -0.1),
        ("prediction_stale_s", float("nan")),
        ("max_packet_bytes", True),
        ("max_packet_bytes", 0),
        ("max_packet_bytes", 1201),
        ("frame_id", ""),
        ("object_name", "  "),
    ],
)
def test_load_config_rejects_invalid_field_values(tmp_path, field, bad_value):
    """Catches missing type/range/network validation on every config field."""
    from netcatch_udp.config import load_config

    values = dict(VALID_CONFIG)
    values[field] = bad_value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("mutation", ["missing", "extra", "non_object"])
def test_load_config_rejects_wrong_shape(tmp_path, mutation):
    """Catches accepting incomplete, extended, or non-object configuration."""
    from netcatch_udp.config import load_config

    values = dict(VALID_CONFIG)
    if mutation == "missing":
        del values["object_name"]
    elif mutation == "extra":
        values["unexpected"] = 1
    else:
        values = [values]
    path = tmp_path / "bad-shape.json"
    path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "literal",
    [
        '{"prediction_topic": NaN}',
        '{"prediction_topic": "/first", "prediction_topic": "/second"}',
    ],
)
def test_load_config_rejects_nonstandard_or_ambiguous_json(tmp_path, literal):
    """Catches Python JSON extensions and duplicate-key ambiguity."""
    from netcatch_udp.config import load_config

    path = tmp_path / "ambiguous.json"
    path.write_text(literal, encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_normalizes_huge_integer_float_overflow(tmp_path):
    """Catches a valid JSON integer escaping config validation as raw OverflowError."""
    from netcatch_udp.config import ConfigError, load_config

    values = dict(VALID_CONFIG)
    values["publish_hz"] = 10**400
    path = tmp_path / "huge-integer.json"
    path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ConfigError, match="finite positive"):
        load_config(path)
