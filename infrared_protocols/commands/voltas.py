"""Voltas air-conditioner IR protocol.

Voltas A/C remotes send a single 80-bit (10-byte) frame, most-significant bit
first, with no header: the frame opens directly on the first bit's mark. Byte 9
is a checksum, the bitwise complement of the sum of the preceding nine bytes.

Frame layout (bit 0 is the least significant bit of each byte):
- Byte 0: horizontal swing, and whether the frame changes it
- Byte 1: mode (bits 3-0), fan speed (bits 7-5)
- Byte 2: vertical swing (bits 2-0), wifi (3), turbo (5), sleep (6), power (7)
- Byte 3: temperature (bits 3-0), econo (6), temperature-button (7)
- Bytes 4-7: on/off timers
- Byte 8: light (bit 5), timer enables (bits 7-6)
- Byte 9: checksum

Only the 122LZF window A/C is modelled, the one model with captured frames to
work from. It has no horizontal swing, so byte 0 is the constant the remote
sends for "do not change horizontal swing".

This encoder covers the fields a thermostat needs: power, mode, temperature,
fan and vertical swing. The remaining fields (wifi, light, turbo, econo, sleep
and the timers) keep the values the remote sends when they are unused.
"""

from enum import IntEnum
from typing import Self, override

from . import Command

MIN_TEMP = 16
MAX_TEMP = 30

# Dry always runs at a fixed temperature and fan speed, whatever the caller asks
# for.
DRY_TEMP = 24

_TEMP_OFFSET = MIN_TEMP

_BIT_MARK = 1026
_ONE_SPACE = 2553
_ZERO_SPACE = 554

_BITS = 80
_STATE_LENGTH = 10

# A receiver skews a bit's mark and space by roughly a fixed number of
# microseconds rather than a fixed proportion, so bits get an absolute
# tolerance. Captured frames stray about 40 µs from nominal, and this still
# keeps the zero and one spaces apart (204-904 vs 2203-2903).
_BIT_TOLERANCE = 350

# Byte 0 holds the horizontal swing bit and, above it, a 7-bit field saying
# whether the frame changes horizontal swing at all. The 122LZF cannot swing
# horizontally, so its remote always sends "no change", which in turn forces the
# swing bit to 1.
_SWING_H_NO_CHANGE = 0b0011001
_BYTE0 = (_SWING_H_NO_CHANGE << 1) | 1

# Bits 5-4 of byte 3 are unidentified but are set in every captured frame.
_BYTE3_UNKNOWN = 0b01 << 4

# Bytes 4-7 carry the timers and byte 8 the light and timer enables. These are
# the values the remote sends with no timer set and the light off.
_IDLE_BYTES = (0x3B, 0x3B, 0x3B, 0x11)
_BYTE8 = 0x00

_SWING_V_ON = 0b111
_SWING_V_OFF = 0b000


class VoltasMode(IntEnum):
    """Operating mode, stored in bits 3-0 of byte 1.

    The remote labels :attr:`COOL` "Chill". :attr:`HEAT` is the same as
    :attr:`COOL` without turbo, econo and sleep.
    """

    FAN = 0b0001
    HEAT = 0b0010
    DRY = 0b0100
    COOL = 0b1000


class VoltasFanSpeed(IntEnum):
    """Fan speed, stored in bits 7-5 of byte 1."""

    HIGH = 0b001
    MEDIUM = 0b010
    LOW = 0b100
    AUTO = 0b111


def _checksum(state: list[int]) -> int:
    """Return the checksum byte: the complement of the sum of the other bytes."""
    return ~sum(state[: _STATE_LENGTH - 1]) & 0xFF


def _is_close(actual: int, expected: int) -> bool:
    """Check if a timing is within tolerance of the expected value."""
    return abs(actual - expected) <= _BIT_TOLERANCE


def _decode_bit(mark: int, space: int) -> int | None:
    """Decode one bit from its mark and space, or None if it matches neither."""
    if not _is_close(mark, _BIT_MARK):
        return None
    if _is_close(space, _ZERO_SPACE):
        return 0
    if _is_close(space, _ONE_SPACE):
        return 1
    return None


class VoltasCommand(Command):
    """Voltas air-conditioner IR command.

    ``temperature`` is required for :attr:`VoltasMode.COOL` and
    :attr:`VoltasMode.HEAT`, and ignored otherwise. :attr:`VoltasFanSpeed.AUTO`
    is unavailable in :attr:`VoltasMode.FAN`.
    """

    mode: VoltasMode
    temperature: int | None
    fan: VoltasFanSpeed
    power: bool
    swing_v: bool

    def __init__(
        self,
        *,
        mode: VoltasMode,
        temperature: int | None = None,
        fan: VoltasFanSpeed = VoltasFanSpeed.AUTO,
        power: bool = True,
        swing_v: bool = False,
        modulation: int = 38000,
    ) -> None:
        """Initialize the Voltas IR command."""
        super().__init__(modulation=modulation)

        if mode in (VoltasMode.COOL, VoltasMode.HEAT):
            if temperature is None:
                raise ValueError(f"temperature is required for mode {mode.name}")
            if not MIN_TEMP <= temperature <= MAX_TEMP:
                raise ValueError(
                    f"temperature {temperature} out of range {MIN_TEMP}..{MAX_TEMP}"
                )

        if mode is VoltasMode.FAN and fan is VoltasFanSpeed.AUTO:
            raise ValueError("fan AUTO is not available in mode FAN")

        self.mode = mode
        # Only cool and heat carry a temperature; storing one the frame cannot
        # express would make a command unequal to itself after a roundtrip.
        self.temperature = (
            temperature if mode in (VoltasMode.COOL, VoltasMode.HEAT) else None
        )
        self.fan = VoltasFanSpeed.LOW if mode is VoltasMode.DRY else fan
        self.power = power
        self.swing_v = swing_v

    def _temperature_nibble(self) -> int:
        """Return the value for byte 3's temperature field."""
        if self.mode is VoltasMode.DRY:
            return DRY_TEMP - _TEMP_OFFSET
        if self.temperature is None:
            # Fan mode ignores the field, but the frame still carries it.
            return 0
        return self.temperature - _TEMP_OFFSET

    def _build_state(self) -> list[int]:
        """Build the 10 state bytes, checksum included."""
        state = [
            _BYTE0,
            self.mode | (self.fan << 5),
            (_SWING_V_ON if self.swing_v else _SWING_V_OFF) | (self.power << 7),
            self._temperature_nibble() | _BYTE3_UNKNOWN,
            *_IDLE_BYTES,
            _BYTE8,
        ]
        state.append(_checksum(state))
        return state

    @override
    def get_raw_timings(self) -> list[int]:
        """Get raw timings for the Voltas command."""
        timings: list[int] = []
        for byte in self._build_state():
            for i in range(7, -1, -1):
                timings.append(_BIT_MARK)
                timings.append(-(_ONE_SPACE if (byte >> i) & 1 else _ZERO_SPACE))
        timings.append(_BIT_MARK)
        return timings

    @classmethod
    def from_raw_timings(cls, timings: list[int]) -> Self | None:
        """Decode raw IR timings into a VoltasCommand.

        Returns a VoltasCommand if the timings match, or None otherwise. Fields
        this class does not model are ignored, so a frame that sets them decodes
        to the state it shares with a frame that does not.
        """
        # 80 bit pairs (160) + the trailing mark (1)
        if len(timings) < 2 * _BITS + 1:
            return None

        state: list[int] = []
        for byte_index in range(_STATE_LENGTH):
            byte = 0
            for bit_index in range(8):
                i = 2 * (byte_index * 8 + bit_index)
                bit = _decode_bit(timings[i], abs(timings[i + 1]))
                if bit is None:
                    return None
                byte = (byte << 1) | bit
            state.append(byte)

        if not _is_close(timings[2 * _BITS], _BIT_MARK):
            return None

        # The frame has no header, so byte 0 and the checksum are all that
        # separate a Voltas frame from another protocol's.
        if state[0] != _BYTE0:
            return None
        if state[_STATE_LENGTH - 1] != _checksum(state):
            return None

        try:
            mode = VoltasMode(state[1] & 0xF)
            fan = VoltasFanSpeed((state[1] >> 5) & 0b111)
        except ValueError:
            return None

        # Checked here rather than left to __init__, which raises where a
        # decoder has to return None.
        if mode is VoltasMode.FAN and fan is VoltasFanSpeed.AUTO:
            return None

        temperature: int | None = None
        if mode in (VoltasMode.COOL, VoltasMode.HEAT):
            temperature = (state[3] & 0xF) + _TEMP_OFFSET
            if not MIN_TEMP <= temperature <= MAX_TEMP:
                return None

        return cls(
            mode=mode,
            temperature=temperature,
            fan=fan,
            power=bool(state[2] >> 7),
            swing_v=(state[2] & 0b111) == _SWING_V_ON,
        )
