"""Tests for the Voltas air-conditioner IR command."""

import pytest

from infrared_protocols.commands.voltas import (
    VoltasCommand,
    VoltasFanSpeed,
    VoltasMode,
)

# Physical-layer constants are duplicated here rather than imported
# so the tests are independent
_BITS = 80
_STATE_LENGTH = 10
_BIT_MARK = 1026
_ONE_SPACE = 2553
_ZERO_SPACE = 554

# Frames captured from a real 122LZF remote.
# The reset state the remote powers up in: cool, 23 °C, high fan, powered off.
_RESET = [0x33, 0x28, 0x00, 0x17, 0x3B, 0x3B, 0x3B, 0x11, 0x00, 0xCB]
# Dry mode, no swing, powered on. Also has wifi set, which is not modelled.
_DRY = [0x33, 0x84, 0x88, 0x18, 0x3B, 0x3B, 0x3B, 0x11, 0x00, 0xE6]
# Cool, 27 °C, high fan, powered on, with an off-timer set. Timers are not
# modelled.
_COOL_27_HIGH_OFF_TIMER = [0x33, 0x28, 0x80, 0x1B, 0x3B, 0x3B, 0x3B, 0x71, 0x40, 0xA7]

_ONE_THRESHOLD = (_ONE_SPACE + _ZERO_SPACE) // 2


def _extract_state(timings: list[int]) -> list[int]:
    """Extract the state bytes from raw timings, one bit per mark/space pair."""
    state: list[int] = []
    for byte_index in range(_STATE_LENGTH):
        byte = 0
        for bit_index in range(8):
            i = 2 * (byte_index * 8 + bit_index) + 1
            byte = (byte << 1) | (1 if abs(timings[i]) > _ONE_THRESHOLD else 0)
        state.append(byte)
    return state


def _build_timings(state: list[int]) -> list[int]:
    """Build raw timings for a state without going through the encoder."""
    timings: list[int] = []
    for byte in state:
        for i in range(7, -1, -1):
            timings.append(_BIT_MARK)
            timings.append(-(_ONE_SPACE if (byte >> i) & 1 else _ZERO_SPACE))
    timings.append(_BIT_MARK)
    return timings


def _resign(state: list[int]) -> list[int]:
    """Return the state with its checksum byte corrected."""
    return [*state[:-1], ~sum(state[:-1]) & 0xFF]


def test_encode_timing_values() -> None:
    """Pin the physical layer: no header, bit mark, and the two bit spaces."""
    timings = VoltasCommand(mode=VoltasMode.COOL, temperature=24).get_raw_timings()

    assert len(timings) == 2 * _BITS + 1
    assert all(mark == _BIT_MARK for mark in timings[::2])
    assert {abs(space) for space in timings[1::2]} == {_ZERO_SPACE, _ONE_SPACE}


def test_encode_matches_captured_reset_frame() -> None:
    """Encoder output must equal the frame the remote sends."""
    cmd = VoltasCommand(
        mode=VoltasMode.COOL,
        temperature=23,
        fan=VoltasFanSpeed.HIGH,
        power=False,
    )
    assert _extract_state(cmd.get_raw_timings()) == _RESET


def test_decode_captured_real_timings() -> None:
    """The raw timings of a real dry-mode capture must decode to its state."""
    # Captured from a 122LZF remote, receiver jitter included.
    raw = [
        1002, 584, 1000, 586, 1000, 2568, 1002, 2570, 1002, 586, 998, 588,
        1000, 2568, 1002, 2570, 1002, 2572, 1002, 584, 1002, 586, 1000, 584,
        1000, 586, 1002, 2568, 1004, 584, 1000, 586, 1002, 2568, 1002, 584,
        1002, 584, 1004, 584, 1000, 2568, 1002, 586, 1000, 586, 998, 590,
        998, 584, 1002, 584, 1000, 586, 1000, 2570, 1002, 2568, 1004, 584,
        1000, 584, 1002, 584, 1002, 582, 1004, 584, 1002, 2568, 1002, 2570,
        1004, 2570, 1000, 586, 1002, 2568, 1004, 2568, 1006, 584, 1000, 584,
        1002, 2568, 1002, 2570, 1002, 2568, 1002, 586, 1002, 2570, 1000, 2570,
        1002, 588, 998, 586, 1000, 2568, 1004, 2568, 1004, 2568, 1002, 588,
        998, 2570, 1002, 2568, 1004, 586, 1002, 584, 1000, 586, 1000, 2570,
        1000, 586, 1000, 584, 1002, 586, 1000, 2568, 1004, 584, 1000, 586,
        1000, 586, 1002, 584, 1002, 586, 1000, 586, 1000, 586, 1000, 586,
        1000, 2568, 1002, 2568, 1002, 2568, 1004, 586, 1000, 584, 1000, 2570,
        1004, 2568, 1004, 584, 1002,
    ]  # fmt: skip
    timings = [v if i % 2 == 0 else -v for i, v in enumerate(raw)]

    result = VoltasCommand.from_raw_timings(timings)
    assert result is not None
    assert result.mode == VoltasMode.DRY
    assert result.fan == VoltasFanSpeed.LOW
    assert result.temperature is None
    assert result.power is True
    assert result.swing_v is False


@pytest.mark.parametrize(
    ("state", "mode", "temperature", "fan", "power"),
    [
        pytest.param(
            _RESET, VoltasMode.COOL, 23, VoltasFanSpeed.HIGH, False, id="reset"
        ),
        pytest.param(_DRY, VoltasMode.DRY, None, VoltasFanSpeed.LOW, True, id="dry"),
        pytest.param(
            _COOL_27_HIGH_OFF_TIMER,
            VoltasMode.COOL,
            27,
            VoltasFanSpeed.HIGH,
            True,
            id="cool_27_high_off_timer",
        ),
    ],
)
def test_decode_captured_frame(
    state: list[int],
    mode: VoltasMode,
    temperature: int | None,
    fan: VoltasFanSpeed,
    power: bool,
) -> None:
    """Captured frames must decode to the state the remote was in."""
    result = VoltasCommand.from_raw_timings(_build_timings(state))
    assert result is not None
    assert result.mode == mode
    assert result.temperature == temperature
    assert result.fan == fan
    assert result.power == power


def test_encode_dry_pins_temperature_and_fan() -> None:
    """Dry runs at 24 °C on a low fan whatever the caller asks for."""
    cmd = VoltasCommand(mode=VoltasMode.DRY, temperature=20, fan=VoltasFanSpeed.HIGH)
    state = _extract_state(cmd.get_raw_timings())

    assert state[3] & 0xF == 24 - 16
    assert (state[1] >> 5) & 0b111 == VoltasFanSpeed.LOW


def test_dry_state_matches_capture_apart_from_wifi() -> None:
    """Dry must encode the captured frame, whose only extra field is wifi."""
    cmd = VoltasCommand(mode=VoltasMode.DRY)
    state = _extract_state(cmd.get_raw_timings())

    state[2] |= 1 << 3
    assert _resign(state) == _DRY


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(VoltasMode.DRY, id="dry"),
        pytest.param(VoltasMode.FAN, id="fan"),
    ],
)
def test_temperature_dropped_for_modes_that_ignore_it(mode: VoltasMode) -> None:
    """A temperature the frame cannot carry must not be stored."""
    cmd = VoltasCommand(mode=mode, temperature=25, fan=VoltasFanSpeed.LOW)
    assert cmd.temperature is None


def test_fan_stored_for_modes_that_carry_it() -> None:
    """A fan speed the caller sets must survive on modes that allow it."""
    cmd = VoltasCommand(mode=VoltasMode.COOL, temperature=24, fan=VoltasFanSpeed.MEDIUM)
    assert cmd.fan == VoltasFanSpeed.MEDIUM


def test_default_modulation() -> None:
    """Default modulation must be 38 kHz, and the protocol has no repeats."""
    cmd = VoltasCommand(mode=VoltasMode.COOL, temperature=24)
    assert cmd.modulation == 38000
    assert cmd.repeat_count == 0


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(VoltasMode.COOL, id="cool"),
        pytest.param(VoltasMode.HEAT, id="heat"),
    ],
)
def test_temperature_required_for_cool_heat(mode: VoltasMode) -> None:
    """COOL and HEAT modes must raise when temperature is omitted."""
    with pytest.raises(ValueError, match="temperature is required"):
        VoltasCommand(mode=mode)


@pytest.mark.parametrize(
    ("temp", "mode"),
    [
        pytest.param(15, VoltasMode.COOL, id="below_min_cool"),
        pytest.param(31, VoltasMode.COOL, id="above_max_cool"),
        pytest.param(15, VoltasMode.HEAT, id="below_min_heat"),
        pytest.param(31, VoltasMode.HEAT, id="above_max_heat"),
    ],
)
def test_temperature_out_of_range(temp: int, mode: VoltasMode) -> None:
    """Out-of-range temperatures must raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        VoltasCommand(mode=mode, temperature=temp)


def test_fan_mode_rejects_auto_fan() -> None:
    """The unit has no auto fan speed in fan mode, so it must not be encoded."""
    with pytest.raises(ValueError, match="AUTO is not available"):
        VoltasCommand(mode=VoltasMode.FAN, fan=VoltasFanSpeed.AUTO)


@pytest.mark.parametrize(
    ("mode", "temperature", "fan", "power", "swing_v"),
    [
        pytest.param(
            VoltasMode.COOL, 16, VoltasFanSpeed.AUTO, True, False, id="cool_min"
        ),
        pytest.param(
            VoltasMode.COOL, 30, VoltasFanSpeed.LOW, True, True, id="cool_max_swing"
        ),
        pytest.param(
            VoltasMode.HEAT, 21, VoltasFanSpeed.MEDIUM, True, False, id="heat_21"
        ),
        pytest.param(
            VoltasMode.HEAT, 24, VoltasFanSpeed.AUTO, False, True, id="heat_off_swing"
        ),
        pytest.param(VoltasMode.DRY, None, VoltasFanSpeed.LOW, True, False, id="dry"),
        pytest.param(
            VoltasMode.FAN, None, VoltasFanSpeed.HIGH, True, True, id="fan_high_swing"
        ),
    ],
)
def test_roundtrip(
    mode: VoltasMode,
    temperature: int | None,
    fan: VoltasFanSpeed,
    power: bool,
    swing_v: bool,
) -> None:
    """Roundtrip encode-decode must preserve every modelled field."""
    cmd = VoltasCommand(
        mode=mode, temperature=temperature, fan=fan, power=power, swing_v=swing_v
    )
    result = VoltasCommand.from_raw_timings(cmd.get_raw_timings())

    assert result is not None
    assert result.mode == mode
    assert result.temperature == temperature
    assert result.fan == fan
    assert result.power == power
    assert result.swing_v == swing_v


def test_decode_returns_none_for_short_timings() -> None:
    """from_raw_timings must return None for incomplete signals."""
    assert VoltasCommand.from_raw_timings([1026, -554]) is None


def test_decode_returns_none_without_trailing_mark() -> None:
    """A frame truncated before its trailing mark must not decode as valid."""
    assert VoltasCommand.from_raw_timings(_build_timings(_RESET)[:-1]) is None


def test_decode_returns_none_for_nec_signal() -> None:
    """A Voltas frame has no header, so another protocol must not decode as one."""
    junk = [9000, -4500] + [560, -1690] * 32 + [560]
    assert VoltasCommand.from_raw_timings(junk) is None


def test_decode_returns_none_for_out_of_tolerance_bit() -> None:
    """A space matching neither a zero nor a one must reject the frame."""
    timings = _build_timings(_RESET)
    timings[1] = -1500  # between the zero (554) and one (2553) spaces
    assert VoltasCommand.from_raw_timings(timings) is None


@pytest.mark.parametrize(
    "state",
    [
        # Byte 9 is 0x00 where 0xCB is correct.
        pytest.param([*_RESET[:-1], 0x00], id="bad_checksum"),
        # Byte 0 identifies the model, and 0xF9 is another Voltas remote's.
        pytest.param(_resign([0xF9, *_RESET[1:]]), id="unknown_byte0"),
        # Mode 0b0011 maps to no VoltasMode.
        pytest.param(_resign([_RESET[0], 0x23, *_RESET[2:]]), id="unknown_mode"),
        # Fan 0b000 maps to no VoltasFanSpeed.
        pytest.param(_resign([_RESET[0], 0x08, *_RESET[2:]]), id="unknown_fan"),
        # Cool with temp nibble 0xF decodes to 31 °C, above MAX_TEMP.
        pytest.param(_resign([*_RESET[:3], 0x1F, *_RESET[4:]]), id="temp_above_max"),
        # Fan mode with an auto fan speed, which the unit has no setting for.
        pytest.param(_resign([_RESET[0], 0xE1, *_RESET[2:]]), id="fan_mode_auto_fan"),
    ],
)
def test_decode_returns_none_for_invalid_frame(state: list[int]) -> None:
    """from_raw_timings must reject frames that fail validation."""
    assert VoltasCommand.from_raw_timings(_build_timings(state)) is None
