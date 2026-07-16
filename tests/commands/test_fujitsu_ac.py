"""Tests for the Fujitsu air-conditioner IR command."""

import pytest

from infrared_protocols.commands.fujitsu_ac import (
    FujitsuAcCommand,
    FujitsuAcFanSpeed,
    FujitsuAcMode,
    FujitsuAcSwing,
)

# Physical-layer constants are duplicated here rather than imported
# so the tests are independent
_HDR_MARK = 3324
_HDR_SPACE = 1574
_BIT_MARK = 448
_ONE_SPACE = 1182
_ZERO_SPACE = 390

_LONG_BITS = 128
_SHORT_BITS = 56

# Frames captured from a real O General unit.
# Powered off.
_OFF = [0x14, 0x63, 0x00, 0x10, 0x10, 0x02, 0xFD]
# Powered on, cool, 24 °C, high fan, no swing.
_COOL_24_HIGH = [
    0x14, 0x63, 0x00, 0x10, 0x10, 0xFE, 0x09, 0x30,
    0x81, 0x01, 0x01, 0x00, 0x00, 0x00, 0x20, 0x2D,
]  # fmt: skip

# Frames captured from a Fujitsu remote of the same variant.
# Cool, 24 °C, high fan, swinging both ways.
_COOL_24_HIGH_BOTH = [
    0x14, 0x63, 0x00, 0x10, 0x10, 0xFE, 0x09, 0x30,
    0x81, 0x01, 0x31, 0x00, 0x00, 0x00, 0x20, 0xFD,
]  # fmt: skip
# Cool, 25 °C, quiet fan, swinging horizontally. Changes a setting on a running
# unit rather than turning it on, so its power bit is clear.
_SETTING_COOL_25_QUIET_HORIZ = [
    0x14, 0x63, 0x00, 0x10, 0x10, 0xFE, 0x09, 0x30,
    0x90, 0x01, 0x24, 0x00, 0x00, 0x00, 0x20, 0xFB,
]  # fmt: skip

_ONE_THRESHOLD = (_ONE_SPACE + _ZERO_SPACE) // 2


def _extract_state(timings: list[int]) -> list[int]:
    """Extract the state bytes from raw timings, one bit per mark/space pair."""
    state: list[int] = []
    for byte_index in range((len(timings) - 3) // 2 // 8):
        byte = 0
        for bit_index in range(8):
            i = 2 + 2 * (byte_index * 8 + bit_index) + 1
            byte |= (1 if abs(timings[i]) > _ONE_THRESHOLD else 0) << bit_index
        state.append(byte)
    return state


def _build_timings(state: list[int]) -> list[int]:
    """Build raw timings for a state without going through the encoder."""
    timings: list[int] = [_HDR_MARK, -_HDR_SPACE]
    for byte in state:
        for i in range(8):
            timings.append(_BIT_MARK)
            timings.append(-(_ONE_SPACE if (byte >> i) & 1 else _ZERO_SPACE))
    timings.append(_BIT_MARK)
    return timings


def _resign_long(state: list[int]) -> list[int]:
    """Return a state frame with its checksum byte corrected."""
    return [*state[:-1], -sum(state[7:15]) & 0xFF]


def test_encode_timing_values() -> None:
    """Pin the physical layer: header, bit mark, and the two bit spaces."""
    timings = FujitsuAcCommand(
        mode=FujitsuAcMode.COOL, temperature=24
    ).get_raw_timings()

    assert timings[:2] == [_HDR_MARK, -_HDR_SPACE]
    assert len(timings) == 2 + 2 * _LONG_BITS + 1
    assert all(mark == _BIT_MARK for mark in timings[2::2])
    assert {abs(space) for space in timings[3::2]} == {_ZERO_SPACE, _ONE_SPACE}


def test_encode_off_is_a_short_frame() -> None:
    """Powering off must send the short frame, not a state frame."""
    timings = FujitsuAcCommand(power=False).get_raw_timings()

    assert len(timings) == 2 + 2 * _SHORT_BITS + 1
    assert _extract_state(timings) == _OFF


def test_encode_is_least_significant_bit_first() -> None:
    """Frames go out least-significant bit first, so byte 0 (0x14) leads 0,0,1."""
    timings = FujitsuAcCommand(power=False).get_raw_timings()
    spaces = [abs(space) for space in timings[3:13:2]]

    assert spaces == [
        _ZERO_SPACE,
        _ZERO_SPACE,
        _ONE_SPACE,
        _ZERO_SPACE,
        _ONE_SPACE,
    ]


@pytest.mark.parametrize(
    ("mode", "temperature", "fan", "swing", "expected_state"),
    [
        pytest.param(
            FujitsuAcMode.COOL,
            24,
            FujitsuAcFanSpeed.HIGH,
            FujitsuAcSwing.OFF,
            _COOL_24_HIGH,
            id="cool_24_high",
        ),
        pytest.param(
            FujitsuAcMode.COOL,
            24,
            FujitsuAcFanSpeed.HIGH,
            FujitsuAcSwing.BOTH,
            _COOL_24_HIGH_BOTH,
            id="cool_24_high_both",
        ),
    ],
)
def test_encode_matches_captured_frame(
    mode: FujitsuAcMode,
    temperature: int,
    fan: FujitsuAcFanSpeed,
    swing: FujitsuAcSwing,
    expected_state: list[int],
) -> None:
    """Encoder output must equal the frame the remote sends."""
    cmd = FujitsuAcCommand(mode=mode, temperature=temperature, fan=fan, swing=swing)
    assert _extract_state(cmd.get_raw_timings()) == expected_state


def test_decode_captured_real_timings() -> None:
    """The raw timings of a real power-off capture must decode to its state."""
    # Captured from an O General unit, receiver jitter included.
    raw = [
        3230, 1660, 450, 398, 540, 370, 478, 1146, 450, 368, 452, 1174,
        450, 368, 452, 368, 478, 346, 478, 1146, 452, 1178, 450, 368,
        476, 342, 478, 340, 478, 1146, 450, 1178, 452, 374, 476, 338,
        478, 338, 478, 340, 478, 414, 478, 340, 478, 340, 478, 340,
        480, 346, 478, 338, 480, 338, 478, 340, 478, 340, 478, 1146,
        452, 368, 452, 368, 478, 348, 478, 336, 478, 340, 478, 340,
        478, 340, 480, 1144, 452, 368, 476, 342, 478, 346, 478, 338,
        478, 1176, 450, 368, 474, 344, 478, 340, 480, 338, 480, 340,
        478, 346, 478, 1144, 452, 366, 476, 1152, 452, 1178, 452, 1176,
        452, 1172, 452, 1178, 450, 1178, 452,
    ]  # fmt: skip
    timings = [v if i % 2 == 0 else -v for i, v in enumerate(raw)]

    result = FujitsuAcCommand.from_raw_timings(timings)
    assert result is not None
    assert result.power is False
    assert result.mode is None
    assert result.temperature is None


@pytest.mark.parametrize(
    ("state", "mode", "temperature", "fan", "swing"),
    [
        pytest.param(
            _COOL_24_HIGH,
            FujitsuAcMode.COOL,
            24,
            FujitsuAcFanSpeed.HIGH,
            FujitsuAcSwing.OFF,
            id="cool_24_high",
        ),
        pytest.param(
            _COOL_24_HIGH_BOTH,
            FujitsuAcMode.COOL,
            24,
            FujitsuAcFanSpeed.HIGH,
            FujitsuAcSwing.BOTH,
            id="cool_24_high_both",
        ),
        pytest.param(
            _SETTING_COOL_25_QUIET_HORIZ,
            FujitsuAcMode.COOL,
            25,
            FujitsuAcFanSpeed.QUIET,
            FujitsuAcSwing.HORIZONTAL,
            id="setting_cool_25_quiet_horiz",
        ),
    ],
)
def test_decode_captured_frame(
    state: list[int],
    mode: FujitsuAcMode,
    temperature: int,
    fan: FujitsuAcFanSpeed,
    swing: FujitsuAcSwing,
) -> None:
    """Turn-on and setting-change frames must both decode as powered on."""
    result = FujitsuAcCommand.from_raw_timings(_build_timings(state))
    assert result is not None
    assert result.power is True
    assert result.mode == mode
    assert result.temperature == temperature
    assert result.fan == fan
    assert result.swing == swing


def test_decode_captured_off_frame() -> None:
    """The captured power-off frame must decode to a powered-off state."""
    result = FujitsuAcCommand.from_raw_timings(_build_timings(_OFF))
    assert result is not None
    assert result.power is False


@pytest.mark.parametrize(
    ("temperature", "expected_field"),
    [
        pytest.param(16, 0, id="min"),
        pytest.param(24, 32, id="mid"),
        pytest.param(30, 56, id="max"),
    ],
)
def test_encode_temperature_scaling(temperature: int, expected_field: int) -> None:
    """The frame stores the temperature in quarter degrees above 16 °C."""
    cmd = FujitsuAcCommand(mode=FujitsuAcMode.COOL, temperature=temperature)
    state = _extract_state(cmd.get_raw_timings())

    assert (state[8] >> 2) & 0x3F == expected_field


def test_off_drops_state_it_cannot_send() -> None:
    """The short frame carries no state, so none must be stored."""
    cmd = FujitsuAcCommand(
        mode=FujitsuAcMode.HEAT,
        temperature=25,
        fan=FujitsuAcFanSpeed.HIGH,
        swing=FujitsuAcSwing.BOTH,
        power=False,
    )

    assert cmd.mode is None
    assert cmd.temperature is None
    assert cmd.fan is None
    assert cmd.swing is None


def test_default_modulation() -> None:
    """Default modulation must be 38 kHz, and the protocol has no repeats."""
    cmd = FujitsuAcCommand(mode=FujitsuAcMode.COOL, temperature=24)
    assert cmd.modulation == 38000
    assert cmd.repeat_count == 0


def test_mode_required_while_power_is_on() -> None:
    """A state frame must carry a mode."""
    with pytest.raises(ValueError, match="mode is required"):
        FujitsuAcCommand(temperature=24)


def test_temperature_required_while_power_is_on() -> None:
    """A state frame always carries a temperature, so one must be given."""
    with pytest.raises(ValueError, match="temperature is required"):
        FujitsuAcCommand(mode=FujitsuAcMode.COOL)


@pytest.mark.parametrize(
    "temperature",
    [
        pytest.param(15, id="below_min"),
        pytest.param(31, id="above_max"),
    ],
)
def test_temperature_out_of_range(temperature: int) -> None:
    """Out-of-range temperatures must raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        FujitsuAcCommand(mode=FujitsuAcMode.COOL, temperature=temperature)


@pytest.mark.parametrize(
    ("mode", "temperature", "fan", "swing", "power"),
    [
        pytest.param(
            FujitsuAcMode.AUTO,
            16,
            FujitsuAcFanSpeed.AUTO,
            FujitsuAcSwing.OFF,
            True,
            id="auto_min",
        ),
        pytest.param(
            FujitsuAcMode.HEAT,
            30,
            FujitsuAcFanSpeed.QUIET,
            FujitsuAcSwing.BOTH,
            True,
            id="heat_max_both",
        ),
        pytest.param(
            FujitsuAcMode.DRY,
            22,
            FujitsuAcFanSpeed.LOW,
            FujitsuAcSwing.VERTICAL,
            True,
            id="dry_vertical",
        ),
        pytest.param(
            FujitsuAcMode.FAN,
            24,
            FujitsuAcFanSpeed.MEDIUM,
            FujitsuAcSwing.HORIZONTAL,
            True,
            id="fan_horizontal",
        ),
        pytest.param(
            None, None, FujitsuAcFanSpeed.AUTO, FujitsuAcSwing.OFF, False, id="off"
        ),
    ],
)
def test_roundtrip(
    mode: FujitsuAcMode | None,
    temperature: int | None,
    fan: FujitsuAcFanSpeed,
    swing: FujitsuAcSwing,
    power: bool,
) -> None:
    """Roundtrip encode-decode must preserve every modelled field."""
    cmd = FujitsuAcCommand(
        mode=mode, temperature=temperature, fan=fan, swing=swing, power=power
    )
    result = FujitsuAcCommand.from_raw_timings(cmd.get_raw_timings())

    assert result is not None
    assert result.power == power
    assert result.mode == cmd.mode
    assert result.temperature == cmd.temperature
    assert result.fan == cmd.fan
    assert result.swing == cmd.swing


def test_decode_returns_none_for_short_timings() -> None:
    """from_raw_timings must return None for incomplete signals."""
    assert FujitsuAcCommand.from_raw_timings([_HDR_MARK, -_HDR_SPACE]) is None


def test_decode_returns_none_without_trailing_mark() -> None:
    """A frame truncated before its trailing mark must not decode as valid."""
    assert FujitsuAcCommand.from_raw_timings(_build_timings(_OFF)[:-1]) is None


def test_decode_returns_none_for_nec_signal() -> None:
    """Another protocol's signal must not decode as a Fujitsu frame."""
    junk = [9000, -4500] + [560, -1690] * 32 + [560]
    assert FujitsuAcCommand.from_raw_timings(junk) is None


def test_decode_returns_none_for_invalid_header() -> None:
    """A signal whose header matches no Fujitsu frame must be rejected."""
    timings = _build_timings(_OFF)
    timings[1] = -500
    assert FujitsuAcCommand.from_raw_timings(timings) is None


def test_decode_returns_none_for_out_of_tolerance_bit() -> None:
    """A space matching neither a zero nor a one must reject the frame."""
    timings = _build_timings(_OFF)
    timings[3] = -780  # between the zero (390) and one (1182) spaces
    assert FujitsuAcCommand.from_raw_timings(timings) is None


def test_decode_accepts_stretched_header_mark() -> None:
    """Receivers stretch marks; the header mark can arrive well over nominal."""
    timings = _build_timings(_COOL_24_HIGH)
    timings[0] = 4800
    result = FujitsuAcCommand.from_raw_timings(timings)
    assert result is not None
    assert result.mode == FujitsuAcMode.COOL


@pytest.mark.parametrize(
    "state",
    [
        # Byte 15 is 0x00 where 0x2D is correct.
        pytest.param([*_COOL_24_HIGH[:-1], 0x00], id="bad_long_checksum"),
        # Byte 6 is not the complement of byte 5.
        pytest.param([*_OFF[:-1], 0x00], id="bad_short_checksum"),
        # Byte 0 is 0x99 where every frame starts 0x14.
        pytest.param(
            _resign_long([0x99, *_COOL_24_HIGH[1:]]), id="unknown_header_byte"
        ),
        # Protocol 0x31 marks the ARREW4E variant, whose frame differs.
        pytest.param(
            _resign_long([*_COOL_24_HIGH[:7], 0x31, *_COOL_24_HIGH[8:]]),
            id="other_variant_protocol",
        ),
        # Command 0xFC marks the ARDB1 and ARJW2 variants.
        pytest.param(
            _resign_long([*_COOL_24_HIGH[:5], 0xFC, *_COOL_24_HIGH[6:]]),
            id="other_variant_command",
        ),
        # Mode 0b111 maps to no FujitsuAcMode.
        pytest.param(
            _resign_long([*_COOL_24_HIGH[:9], 0x07, *_COOL_24_HIGH[10:]]),
            id="unknown_mode",
        ),
        # Fan 0b111 maps to no FujitsuAcFanSpeed.
        pytest.param(
            _resign_long([*_COOL_24_HIGH[:10], 0x07, *_COOL_24_HIGH[11:]]),
            id="unknown_fan",
        ),
        # Temperature field 60 decodes to 31 °C, above MAX_TEMP.
        pytest.param(
            _resign_long([*_COOL_24_HIGH[:8], 0xF1, *_COOL_24_HIGH[9:]]),
            id="temp_above_max",
        ),
        # The Fahrenheit bit belongs to a variant this class does not model.
        pytest.param(
            _resign_long([*_COOL_24_HIGH[:8], 0x83, *_COOL_24_HIGH[9:]]),
            id="fahrenheit_set",
        ),
        # A short frame that is not the power-off command.
        pytest.param([*_OFF[:5], 0x09, 0xF6], id="other_short_command"),
    ],
)
def test_decode_returns_none_for_invalid_frame(state: list[int]) -> None:
    """from_raw_timings must reject frames that fail validation."""
    assert FujitsuAcCommand.from_raw_timings(_build_timings(state)) is None
