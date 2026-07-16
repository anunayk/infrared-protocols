"""Midea air-conditioner IR protocol.

Midea builds air conditioners for a long list of brands, so the same protocol
drives units badged Midea, Comfee, Danby, Trotec, Pioneer, Lennox, Kaysun,
Keystone and MrCool, among others.

A message is six bytes sent most-significant bit first, then sent a second time
with every bit inverted. The unit only accepts a message whose two halves are
complements, which makes the pair its own integrity check.

Byte layout (bit 0 is the least significant bit of each byte):
- Byte 0: message type; 0xA1 for the control message this module sends
- Byte 1: mode (bits 2-0), fan (bits 4-3), sleep (bit 6), power (bit 7)
- Byte 2: temperature (bits 4-0), Fahrenheit (bit 5)
- Byte 3: off timer
- Byte 4: sensor temperature, or the on timer
- Byte 5: checksum

The unit takes its setpoint in whichever scale byte 2 selects, so ``fahrenheit``
picks the scale the frame carries rather than only what the unit displays.

The A/C also has special messages, sent with a different type byte, that toggle
vertical swing, econo, turbo, light, self clean and 8 °C heat, and that set
quiet mode. None are encoded here. There is no swing setting to read back or
write, only a toggle, and the toggles differ between units: the code that
toggles swing on a Kaysun is the one that toggles econo everywhere else.
"""

from enum import IntEnum
from typing import Self, override

from . import Command

MIN_TEMP = 17
MAX_TEMP = 30
MIN_TEMP_F = 62
MAX_TEMP_F = 86

_TEMP_OFFSET = MIN_TEMP
_TEMP_OFFSET_F = MIN_TEMP_F
_TEMP_MASK = 0b11111

# Every timing is a multiple of a 560 µs tick.
_HDR_MARK = 4480
_HDR_SPACE = 4480
_BIT_MARK = 560
_ZERO_SPACE = 560
_ONE_SPACE = 1680
_GAP = 5600

_BITS = 48
_STATE_LENGTH = 6

# A header pair, the data bits, and the trailing mark.
_PHASE_LENGTH = 2 + 2 * _BITS + 1
# Both halves of the message, with the gap between them.
_TIMING_COUNT = 2 * _PHASE_LENGTH + 1

# Byte 0 holds a fixed 0b10100 above a 3-bit message type. The other types carry
# settings this module does not model.
_TYPE_CONTROL = 0xA1

_FAHRENHEIT_BIT = 0x20
# Bit 6 of byte 2 is unidentified but is set in every captured frame.
_BYTE2_UNKNOWN = 0x40
# Bytes 3 and 4 with no timer set and no sensor temperature reported.
_UNUSED = 0xFF

# IR receivers distort marks (AGC) but keep spaces accurate.
_MARK_TOLERANCE = 0.7
_SPACE_TOLERANCE = 0.25

# A receiver skews a bit's mark and space by roughly a fixed number of
# microseconds rather than a fixed proportion, so bits get an absolute
# tolerance. Captured frames stray about 60 µs from nominal, and this still
# keeps the zero and one spaces apart (210-910 vs 1330-2030).
_BIT_TOLERANCE = 350


class MideaMode(IntEnum):
    """Operating mode, stored in bits 2-0 of byte 1."""

    COOL = 0b000
    DRY = 0b001
    AUTO = 0b010
    HEAT = 0b011
    FAN = 0b100


class MideaFanSpeed(IntEnum):
    """Fan speed, stored in bits 4-3 of byte 1."""

    AUTO = 0b00
    LOW = 0b01
    MEDIUM = 0b10
    HIGH = 0b11


def _reverse_bits(value: int) -> int:
    """Return a byte with its bits in the opposite order."""
    return int(f"{value:08b}"[::-1], 2)


def _checksum(state: list[int]) -> int:
    """Return the checksum byte for a message.

    The bytes are summed and negated with their bit order reversed, and the
    result is reversed back.
    """
    total = sum(_reverse_bits(byte) for byte in state[: _STATE_LENGTH - 1])
    return _reverse_bits(-total & 0xFF)


def _is_close(actual: int, expected: int) -> bool:
    """Check if a bit timing is within tolerance of the expected value."""
    return abs(actual - expected) <= _BIT_TOLERANCE


def _matches_space(actual: int, expected: int) -> bool:
    """Check if a long space is within tolerance of the expected value."""
    return abs(actual - expected) <= expected * _SPACE_TOLERANCE


def _matches_header(mark: int, space: int) -> bool:
    """Check if a mark and space pair is a header."""
    return abs(mark - _HDR_MARK) <= _HDR_MARK * _MARK_TOLERANCE and _matches_space(
        space, _HDR_SPACE
    )


def _decode_bit(mark: int, space: int) -> int | None:
    """Decode one bit from its mark and space, or None if it matches neither."""
    if not _is_close(mark, _BIT_MARK):
        return None
    if _is_close(space, _ZERO_SPACE):
        return 0
    if _is_close(space, _ONE_SPACE):
        return 1
    return None


def _decode_half(timings: list[int], start: int) -> list[int] | None:
    """Decode one half of a message: a header, the bytes, and a trailing mark."""
    if not _matches_header(timings[start], abs(timings[start + 1])):
        return None

    state: list[int] = []
    for byte_index in range(_STATE_LENGTH):
        byte = 0
        for bit_index in range(8):
            i = start + 2 + 2 * (byte_index * 8 + bit_index)
            bit = _decode_bit(timings[i], abs(timings[i + 1]))
            if bit is None:
                return None
            byte = (byte << 1) | bit
        state.append(byte)

    if not _is_close(timings[start + 2 + 2 * _BITS], _BIT_MARK):
        return None
    return state


class MideaCommand(Command):
    """Midea air-conditioner IR command.

    ``temperature`` is in degrees Celsius, or Fahrenheit when ``fahrenheit`` is
    set, and must be within the range the chosen scale allows.
    """

    mode: MideaMode
    temperature: int
    fan: MideaFanSpeed
    power: bool
    fahrenheit: bool

    def __init__(
        self,
        *,
        mode: MideaMode,
        temperature: int,
        fan: MideaFanSpeed = MideaFanSpeed.AUTO,
        power: bool = True,
        fahrenheit: bool = False,
        modulation: int = 38000,
    ) -> None:
        """Initialize the Midea IR command."""
        super().__init__(modulation=modulation)

        min_temp, max_temp = (
            (MIN_TEMP_F, MAX_TEMP_F) if fahrenheit else (MIN_TEMP, MAX_TEMP)
        )
        if not min_temp <= temperature <= max_temp:
            raise ValueError(
                f"temperature {temperature} out of range {min_temp}..{max_temp}"
            )

        self.mode = mode
        self.temperature = temperature
        self.fan = fan
        self.power = power
        self.fahrenheit = fahrenheit

    def _build_state(self) -> list[int]:
        """Build the six message bytes, checksum included."""
        offset = _TEMP_OFFSET_F if self.fahrenheit else _TEMP_OFFSET
        state = [
            _TYPE_CONTROL,
            self.mode | (self.fan << 3) | (self.power << 7),
            (self.temperature - offset)
            | (_FAHRENHEIT_BIT if self.fahrenheit else 0)
            | _BYTE2_UNKNOWN,
            _UNUSED,
            _UNUSED,
        ]
        state.append(_checksum(state))
        return state

    @override
    def get_raw_timings(self) -> list[int]:
        """Get raw timings for the Midea command."""
        state = self._build_state()
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

    @classmethod
    def from_raw_timings(cls, timings: list[int]) -> Self | None:
        """Decode raw IR timings into a MideaCommand.

        Returns a MideaCommand if the timings match, or None otherwise. Settings
        this class does not model are ignored, so a frame that sets one decodes
        to the state it shares with a frame that does not.
        """
        if len(timings) != _TIMING_COUNT:
            return None

        state = _decode_half(timings, 0)
        if state is None:
            return None
        if not _matches_space(abs(timings[_PHASE_LENGTH]), _GAP):
            return None
        inverted = _decode_half(timings, _PHASE_LENGTH + 1)
        if inverted is None:
            return None
        # The unit only accepts a message whose halves are complements.
        if any(a + b != 0xFF for a, b in zip(state, inverted, strict=True)):
            return None

        if state[0] != _TYPE_CONTROL:
            return None
        if state[-1] != _checksum(state):
            return None

        fahrenheit = bool(state[2] & _FAHRENHEIT_BIT)
        offset = _TEMP_OFFSET_F if fahrenheit else _TEMP_OFFSET
        min_temp, max_temp = (
            (MIN_TEMP_F, MAX_TEMP_F) if fahrenheit else (MIN_TEMP, MAX_TEMP)
        )
        temperature = (state[2] & _TEMP_MASK) + offset
        # Checked here rather than left to __init__, which raises where a
        # decoder has to return None.
        if not min_temp <= temperature <= max_temp:
            return None

        try:
            mode = MideaMode(state[1] & 0b111)
            fan = MideaFanSpeed((state[1] >> 3) & 0b11)
        except ValueError:
            return None

        return cls(
            mode=mode,
            temperature=temperature,
            fan=fan,
            power=bool(state[1] >> 7),
            fahrenheit=fahrenheit,
        )
