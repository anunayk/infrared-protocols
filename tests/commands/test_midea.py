"""Tests for the Midea air-conditioner IR command."""

import pytest

from infrared_protocols.commands.midea import (
    MideaCommand,
    MideaFanSpeed,
    MideaMode,
)

# Physical-layer constants are duplicated here rather than imported
# so the tests are independent
_HDR_MARK = 4480
_HDR_SPACE = 4480
_BIT_MARK = 560
_ZERO_SPACE = 560
_ONE_SPACE = 1680
_GAP = 5600

_BITS = 48
_STATE_LENGTH = 6
_PHASE_LENGTH = 2 + 2 * _BITS + 1
_TIMING_COUNT = 2 * _PHASE_LENGTH + 1

# Messages captured from a real Comfee remote, which sends Celsius.
_COOL_LOW_24C = [0xA1, 0x88, 0x47, 0xFF, 0xFF, 0x51]
_COOL_LOW_25C = [0xA1, 0x88, 0x48, 0xFF, 0xFF, 0x5A]
# Celsius range ends, from the same report.
_COOL_LOW_17C = [0xA1, 0x88, 0x40, 0xFF, 0xFF, 0x56]
_COOL_LOW_30C = [0xA1, 0x88, 0x4D, 0xFF, 0xFF, 0x5D]

# Messages captured from a unit set to Fahrenheit.
_AUTO_65F = [0xA1, 0x82, 0x63, 0xFF, 0xFF, 0x6E]
_AUTO_77F = [0xA1, 0x82, 0x6F, 0xFF, 0xFF, 0x62]
_OFF_77F = [0xA1, 0x02, 0x6F, 0xFF, 0xFF, 0xE2]

# A special message: toggles vertical swing rather than setting state.
_TOGGLE_SWING_V = [0xA2, 0x01, 0xFF, 0xFF, 0xFF, 0x7C]

_ONE_THRESHOLD = (_ONE_SPACE + _ZERO_SPACE) // 2


def _extract_state(timings: list[int], start: int = 0) -> list[int]:
    """Extract the message bytes from one half of the raw timings."""
    state: list[int] = []
    for byte_index in range(_STATE_LENGTH):
        byte = 0
        for bit_index in range(8):
            i = start + 2 + 2 * (byte_index * 8 + bit_index) + 1
            byte = (byte << 1) | (1 if abs(timings[i]) > _ONE_THRESHOLD else 0)
        state.append(byte)
    return state


def _build_timings(state: list[int]) -> list[int]:
    """Build raw timings for a message without going through the encoder."""
    timings: list[int] = []
    for inverted in (False, True):
        timings.extend([_HDR_MARK, -_HDR_SPACE])
        for byte in state:
            value = byte ^ 0xFF if inverted else byte
            for i in range(7, -1, -1):
                timings.append(_BIT_MARK)
                timings.append(-(_ONE_SPACE if (value >> i) & 1 else _ZERO_SPACE))
        timings.append(_BIT_MARK)
        if not inverted:
            timings.append(-_GAP)
    return timings


def _resign(state: list[int]) -> list[int]:
    """Return a message with its checksum byte corrected."""

    def reverse(value: int) -> int:
        return int(f"{value:08b}"[::-1], 2)

    total = sum(reverse(byte) for byte in state[:5])
    return [*state[:-1], reverse(-total & 0xFF)]


def test_encode_timing_values() -> None:
    """Pin the physical layer: header, bit mark, and the two bit spaces."""
    timings = MideaCommand(mode=MideaMode.COOL, temperature=24).get_raw_timings()

    assert len(timings) == _TIMING_COUNT
    assert timings[:2] == [_HDR_MARK, -_HDR_SPACE]
    assert timings[_PHASE_LENGTH] == -_GAP
    assert timings[_PHASE_LENGTH + 1 : _PHASE_LENGTH + 3] == [_HDR_MARK, -_HDR_SPACE]
    assert timings[-1] == _BIT_MARK


def test_encode_sends_the_message_twice_inverted() -> None:
    """The second half must be the first with every bit inverted."""
    timings = MideaCommand(mode=MideaMode.COOL, temperature=24).get_raw_timings()

    first = _extract_state(timings)
    second = _extract_state(timings, _PHASE_LENGTH + 1)
    assert all(a + b == 0xFF for a, b in zip(first, second, strict=True))


def test_encode_is_most_significant_bit_first() -> None:
    """Bytes go out most-significant bit first, so byte 0 (0xA1) leads 1,0,1."""
    timings = MideaCommand(mode=MideaMode.COOL, temperature=24).get_raw_timings()
    spaces = [abs(space) for space in timings[3:13:2]]

    assert spaces == [
        _ONE_SPACE,
        _ZERO_SPACE,
        _ONE_SPACE,
        _ZERO_SPACE,
        _ZERO_SPACE,
    ]


@pytest.mark.parametrize(
    ("mode", "temperature", "fan", "power", "fahrenheit", "expected_state"),
    [
        pytest.param(
            MideaMode.COOL,
            24,
            MideaFanSpeed.LOW,
            True,
            False,
            _COOL_LOW_24C,
            id="cool_low_24c",
        ),
        pytest.param(
            MideaMode.COOL,
            25,
            MideaFanSpeed.LOW,
            True,
            False,
            _COOL_LOW_25C,
            id="cool_low_25c",
        ),
        pytest.param(
            MideaMode.COOL,
            17,
            MideaFanSpeed.LOW,
            True,
            False,
            _COOL_LOW_17C,
            id="cool_low_17c_min",
        ),
        pytest.param(
            MideaMode.COOL,
            30,
            MideaFanSpeed.LOW,
            True,
            False,
            _COOL_LOW_30C,
            id="cool_low_30c_max",
        ),
        pytest.param(
            MideaMode.AUTO,
            65,
            MideaFanSpeed.AUTO,
            True,
            True,
            _AUTO_65F,
            id="auto_65f",
        ),
        pytest.param(
            MideaMode.AUTO,
            77,
            MideaFanSpeed.AUTO,
            True,
            True,
            _AUTO_77F,
            id="auto_77f",
        ),
        pytest.param(
            MideaMode.AUTO,
            77,
            MideaFanSpeed.AUTO,
            False,
            True,
            _OFF_77F,
            id="off_77f",
        ),
    ],
)
def test_encode_matches_captured_message(
    mode: MideaMode,
    temperature: int,
    fan: MideaFanSpeed,
    power: bool,
    fahrenheit: bool,
    expected_state: list[int],
) -> None:
    """Encoder output must equal the message the remote sends."""
    cmd = MideaCommand(
        mode=mode,
        temperature=temperature,
        fan=fan,
        power=power,
        fahrenheit=fahrenheit,
    )
    assert _extract_state(cmd.get_raw_timings()) == expected_state


def test_decode_captured_real_timings() -> None:
    """The raw timings of a real capture must decode to its state."""
    # Captured from a unit set to Fahrenheit, receiver jitter included.
    raw = [
        4366, 4470, 498, 1658, 522, 554, 498, 1658, 496, 580, 498, 580,
        498, 578, 498, 580, 498, 1658, 498, 1658, 498, 578, 498, 578,
        498, 580, 496, 582, 496, 578, 498, 1658, 498, 580, 498, 580,
        498, 1656, 498, 1656, 500, 580, 498, 578, 502, 576, 500, 1656,
        498, 1656, 500, 1654, 500, 1656, 500, 1656, 498, 1658, 498, 1656,
        500, 1658, 498, 1656, 498, 1656, 500, 1656, 500, 1654, 500, 1578,
        578, 1658, 498, 1656, 500, 1658, 498, 1656, 498, 1656, 500, 578,
        498, 1638, 516, 1656, 500, 578, 500, 1656, 500, 1656, 498, 1658,
        522, 554, 500, 5258, 4366, 4472, 498, 580, 498, 1658, 498, 580,
        498, 1656, 500, 1600, 556, 1658, 500, 1656, 500, 578, 498, 578,
        522, 1634, 498, 1588, 568, 1658, 498, 1656, 500, 1654, 498, 580,
        498, 1658, 498, 1658, 498, 580, 496, 578, 500, 1654, 500, 1636,
        518, 1656, 500, 578, 520, 558, 498, 578, 498, 580, 498, 576,
        500, 578, 498, 580, 498, 578, 498, 578, 498, 580, 498, 578,
        498, 580, 498, 580, 520, 556, 498, 580, 496, 580, 498, 578,
        500, 578, 498, 1658, 498, 580, 498, 578, 498, 1656, 500, 578,
        498, 580, 498, 580, 498, 1656, 522,
    ]  # fmt: skip
    timings = [v if i % 2 == 0 else -v for i, v in enumerate(raw)]

    result = MideaCommand.from_raw_timings(timings)
    assert result is not None
    assert result.mode == MideaMode.AUTO
    assert result.fan == MideaFanSpeed.AUTO
    assert result.temperature == 65
    assert result.fahrenheit is True
    assert result.power is True


@pytest.mark.parametrize(
    ("state", "mode", "temperature", "fan", "power", "fahrenheit"),
    [
        pytest.param(
            _COOL_LOW_24C,
            MideaMode.COOL,
            24,
            MideaFanSpeed.LOW,
            True,
            False,
            id="cool_low_24c",
        ),
        pytest.param(
            _COOL_LOW_30C,
            MideaMode.COOL,
            30,
            MideaFanSpeed.LOW,
            True,
            False,
            id="cool_low_30c",
        ),
        pytest.param(
            _AUTO_65F,
            MideaMode.AUTO,
            65,
            MideaFanSpeed.AUTO,
            True,
            True,
            id="auto_65f",
        ),
        pytest.param(
            _OFF_77F,
            MideaMode.AUTO,
            77,
            MideaFanSpeed.AUTO,
            False,
            True,
            id="off_77f",
        ),
    ],
)
def test_decode_captured_message(
    state: list[int],
    mode: MideaMode,
    temperature: int,
    fan: MideaFanSpeed,
    power: bool,
    fahrenheit: bool,
) -> None:
    """Captured messages must decode to the state the remote was in."""
    result = MideaCommand.from_raw_timings(_build_timings(state))
    assert result is not None
    assert result.mode == mode
    assert result.temperature == temperature
    assert result.fan == fan
    assert result.power == power
    assert result.fahrenheit == fahrenheit


@pytest.mark.parametrize(
    ("temperature", "fahrenheit", "expected_field"),
    [
        pytest.param(17, False, 0, id="min_celsius"),
        pytest.param(24, False, 7, id="mid_celsius"),
        pytest.param(30, False, 13, id="max_celsius"),
        pytest.param(62, True, 0, id="min_fahrenheit"),
        pytest.param(86, True, 24, id="max_fahrenheit"),
    ],
)
def test_encode_temperature_scaling(
    temperature: int, fahrenheit: bool, expected_field: int
) -> None:
    """The message stores the temperature as an offset from the scale's minimum."""
    cmd = MideaCommand(
        mode=MideaMode.COOL, temperature=temperature, fahrenheit=fahrenheit
    )
    state = _extract_state(cmd.get_raw_timings())

    assert state[2] & 0b11111 == expected_field
    assert bool(state[2] & 0x20) is fahrenheit


def test_default_modulation() -> None:
    """Default modulation must be 38 kHz, and the protocol has no repeats."""
    cmd = MideaCommand(mode=MideaMode.COOL, temperature=24)
    assert cmd.modulation == 38000
    assert cmd.repeat_count == 0


@pytest.mark.parametrize(
    ("temperature", "fahrenheit"),
    [
        pytest.param(16, False, id="below_min_celsius"),
        pytest.param(31, False, id="above_max_celsius"),
        pytest.param(61, True, id="below_min_fahrenheit"),
        pytest.param(87, True, id="above_max_fahrenheit"),
        pytest.param(24, True, id="celsius_value_while_fahrenheit"),
        pytest.param(70, False, id="fahrenheit_value_while_celsius"),
    ],
)
def test_temperature_out_of_range(temperature: int, fahrenheit: bool) -> None:
    """Out-of-range temperatures must raise ValueError for the chosen scale."""
    with pytest.raises(ValueError, match="out of range"):
        MideaCommand(
            mode=MideaMode.COOL, temperature=temperature, fahrenheit=fahrenheit
        )


@pytest.mark.parametrize(
    ("mode", "temperature", "fan", "power", "fahrenheit"),
    [
        pytest.param(
            MideaMode.COOL, 17, MideaFanSpeed.AUTO, True, False, id="cool_min_c"
        ),
        pytest.param(
            MideaMode.HEAT, 30, MideaFanSpeed.HIGH, True, False, id="heat_max_c"
        ),
        pytest.param(MideaMode.DRY, 22, MideaFanSpeed.LOW, True, False, id="dry_c"),
        pytest.param(
            MideaMode.FAN, 25, MideaFanSpeed.MEDIUM, True, False, id="fan_only_c"
        ),
        pytest.param(MideaMode.AUTO, 20, MideaFanSpeed.AUTO, False, False, id="off_c"),
        pytest.param(
            MideaMode.COOL, 62, MideaFanSpeed.HIGH, True, True, id="cool_min_f"
        ),
        pytest.param(
            MideaMode.HEAT, 86, MideaFanSpeed.LOW, True, True, id="heat_max_f"
        ),
    ],
)
def test_roundtrip(
    mode: MideaMode,
    temperature: int,
    fan: MideaFanSpeed,
    power: bool,
    fahrenheit: bool,
) -> None:
    """Roundtrip encode-decode must preserve every modelled field."""
    cmd = MideaCommand(
        mode=mode,
        temperature=temperature,
        fan=fan,
        power=power,
        fahrenheit=fahrenheit,
    )
    result = MideaCommand.from_raw_timings(cmd.get_raw_timings())

    assert result is not None
    assert result.mode == mode
    assert result.temperature == temperature
    assert result.fan == fan
    assert result.power == power
    assert result.fahrenheit == fahrenheit


def test_decode_returns_none_for_short_timings() -> None:
    """from_raw_timings must return None for incomplete signals."""
    assert MideaCommand.from_raw_timings([_HDR_MARK, -_HDR_SPACE]) is None


def test_decode_returns_none_without_trailing_mark() -> None:
    """A message truncated before its trailing mark must not decode as valid."""
    assert MideaCommand.from_raw_timings(_build_timings(_COOL_LOW_24C)[:-1]) is None


def test_decode_returns_none_for_half_a_message() -> None:
    """The inverted half is required, so half a message must not decode."""
    timings = _build_timings(_COOL_LOW_24C)[:_PHASE_LENGTH]
    assert MideaCommand.from_raw_timings(timings) is None


def test_decode_returns_none_for_nec_signal() -> None:
    """Another protocol's signal must not decode as a Midea message."""
    junk = [9000, -4500] + [560, -1690] * 32 + [560]
    assert MideaCommand.from_raw_timings(junk) is None


def test_decode_returns_none_for_invalid_header() -> None:
    """A signal whose header matches no Midea message must be rejected."""
    timings = _build_timings(_COOL_LOW_24C)
    timings[1] = -500
    assert MideaCommand.from_raw_timings(timings) is None


def test_decode_returns_none_for_invalid_gap() -> None:
    """The halves are separated by a gap, which must be the right length."""
    timings = _build_timings(_COOL_LOW_24C)
    timings[_PHASE_LENGTH] = -500
    assert MideaCommand.from_raw_timings(timings) is None


def test_decode_returns_none_for_out_of_tolerance_bit() -> None:
    """A space matching neither a zero nor a one must reject the message."""
    timings = _build_timings(_COOL_LOW_24C)
    timings[3] = -1100  # between the zero (560) and one (1680) spaces
    assert MideaCommand.from_raw_timings(timings) is None


def test_decode_returns_none_when_halves_are_not_complements() -> None:
    """A message whose halves disagree must be rejected."""
    timings = _build_timings(_COOL_LOW_24C)
    # Flip one bit of the second half's first byte.
    timings[_PHASE_LENGTH + 4] = -_ONE_SPACE
    assert MideaCommand.from_raw_timings(timings) is None


def test_decode_accepts_stretched_header_mark() -> None:
    """Receivers stretch marks; the header mark can arrive well over nominal."""
    timings = _build_timings(_COOL_LOW_24C)
    timings[0] = 6000
    timings[_PHASE_LENGTH + 1] = 6000
    result = MideaCommand.from_raw_timings(timings)
    assert result is not None
    assert result.temperature == 24


@pytest.mark.parametrize(
    "state",
    [
        # Byte 5 is 0x00 where 0x51 is correct.
        pytest.param([*_COOL_LOW_24C[:-1], 0x00], id="bad_checksum"),
        # Toggles vertical swing; its bytes do not mean what a control's do.
        pytest.param(_TOGGLE_SWING_V, id="special_toggle_message"),
        # A follow-me message, which reports a room temperature.
        pytest.param(_resign([0xA4, *_COOL_LOW_24C[1:]]), id="follow_me_message"),
        # Mode 0b101 maps to no MideaMode.
        pytest.param(
            _resign([_COOL_LOW_24C[0], 0x8D, *_COOL_LOW_24C[2:]]), id="unknown_mode"
        ),
        # Temperature field 31 decodes to 48 °C, above MAX_TEMP.
        pytest.param(
            _resign([*_COOL_LOW_24C[:2], 0x5F, *_COOL_LOW_24C[3:]]),
            id="temp_above_max_celsius",
        ),
        # Temperature field 25 decodes to 87 °F, above MAX_TEMP_F.
        pytest.param(
            _resign([*_COOL_LOW_24C[:2], 0x79, *_COOL_LOW_24C[3:]]),
            id="temp_above_max_fahrenheit",
        ),
    ],
)
def test_decode_returns_none_for_invalid_message(state: list[int]) -> None:
    """from_raw_timings must reject messages that fail validation."""
    assert MideaCommand.from_raw_timings(_build_timings(state)) is None
