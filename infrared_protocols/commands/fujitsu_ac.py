"""Fujitsu air-conditioner IR protocol.

Fujitsu General sells the same units under the Fujitsu and General brands, and
as O General in India, the Middle East and the GCC, so one protocol covers them
all. This module implements the ARRAH2E variant, which is what the O General
AR-RCL1E remote and the Fujitsu AR-RAH2E, AR-RAC1E, AR-RAE1E and AR-RCE1E
remotes send.

.. warning::
   Only use this on a unit whose remote is known to be ARRAH2E compatible. A
   Fujitsu unit that expects another variant can lock up when it is sent a
   swing setting meant for a different one, and recovers only by being
   physically powered off. Other variants exist (ARDB1, ARJW2, ARREB1E, ARRY4
   and ARREW4E) and differ in frame length, checksum and temperature scaling;
   none of them are encoded here.

Frames are sent least-significant bit first, and come in two lengths:

- A 16-byte state frame carrying power, mode, temperature, fan and swing. Its
  last byte is a checksum: the negated sum of bytes 7 to 14.
- A 7-byte frame that turns the unit off and carries nothing else. Its last
  byte is the complement of the byte before it.

Both open with the same five fixed bytes, and byte 5 says which frame it is.

Layout of the state frame (bit 0 is the least significant bit of each byte):
- Bytes 0-4: fixed header
- Byte 5: command; 0xFE for a state frame
- Byte 6: number of bytes after this one
- Byte 7: protocol version
- Byte 8: power (bit 0), Fahrenheit (1), temperature (bits 7-2)
- Byte 9: mode (bits 2-0), clean (3), timer type (bits 5-4)
- Byte 10: fan (bits 2-0), swing (bits 5-4)
- Bytes 11-13: on/off timers
- Byte 14: filter, outside quiet
- Byte 15: checksum
"""

from enum import IntEnum
from typing import Self, override

from . import Command

MIN_TEMP = 16
MAX_TEMP = 30

_TEMP_OFFSET = MIN_TEMP
# The frame stores the temperature in quarter degrees, though this variant only
# ever sets whole ones.
_TEMP_STEP = 4

_HDR_MARK = 3324
_HDR_SPACE = 1574
_BIT_MARK = 448
_ONE_SPACE = 1182
_ZERO_SPACE = 390

# IR receivers distort marks (AGC) but keep spaces accurate.
_MARK_TOLERANCE = 0.7
_SPACE_TOLERANCE = 0.25

# A receiver skews a bit's mark and space by roughly a fixed number of
# microseconds rather than a fixed proportion, so bits get an absolute
# tolerance. Captured frames stray about 50 µs from nominal, and this still
# keeps the zero and one spaces apart (40-740 vs 832-1532).
_BIT_TOLERANCE = 350

_HEADER = [0x14, 0x63, 0x00, 0x10, 0x10]

_CMD_INDEX = 5
_CMD_STATE = 0xFE
_CMD_TURN_OFF = 0x02

_PROTOCOL_INDEX = 7
_PROTOCOL = 0x30

_STATE_LENGTH = 16
_SHORT_STATE_LENGTH = 7
_LONG_BITS = _STATE_LENGTH * 8
_SHORT_BITS = _SHORT_STATE_LENGTH * 8

# Byte 6 counts the bytes after itself.
_REST_LENGTH = _STATE_LENGTH - 7

# The checksum covers the bytes from where a short frame would have ended up to
# the byte before the checksum itself.
_CHECKSUM_START = _SHORT_STATE_LENGTH

_POWER_BIT = 0x01
_FAHRENHEIT_BIT = 0x02

# Byte 14 is unidentified but is set in every captured frame of this variant.
_BYTE14 = 0x20


class FujitsuAcMode(IntEnum):
    """Operating mode, stored in bits 2-0 of byte 9."""

    AUTO = 0x0
    COOL = 0x1
    DRY = 0x2
    FAN = 0x3
    HEAT = 0x4


class FujitsuAcFanSpeed(IntEnum):
    """Fan speed, stored in bits 2-0 of byte 10."""

    AUTO = 0x0
    HIGH = 0x1
    MEDIUM = 0x2
    LOW = 0x3
    QUIET = 0x4


class FujitsuAcSwing(IntEnum):
    """Swing setting, stored in bits 5-4 of byte 10.

    See the module warning: a unit that expects another variant can lock up on a
    swing setting meant for a different one.
    """

    OFF = 0b00
    VERTICAL = 0b01
    HORIZONTAL = 0b10
    BOTH = 0b11


def _long_checksum(state: list[int]) -> int:
    """Return the checksum byte: the negated sum of the payload bytes."""
    return -sum(state[_CHECKSUM_START : _STATE_LENGTH - 1]) & 0xFF


def _is_close(actual: int, expected: int) -> bool:
    """Check if a bit timing is within tolerance of the expected value."""
    return abs(actual - expected) <= _BIT_TOLERANCE


def _matches_header(mark: int, space: int) -> bool:
    """Check if a mark and space pair is the header."""
    return (
        abs(mark - _HDR_MARK) <= _HDR_MARK * _MARK_TOLERANCE
        and abs(space - _HDR_SPACE) <= _HDR_SPACE * _SPACE_TOLERANCE
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


class FujitsuAcCommand(Command):
    """Fujitsu air-conditioner IR command, as used by O General units.

    ``mode`` and ``temperature`` are required while ``power`` is on, and ignored
    when it is off, because the frame that turns the unit off carries no state.
    ``FujitsuAcMode.FAN`` still needs a temperature: the frame always carries
    one, even though the unit ignores it in that mode.
    """

    mode: FujitsuAcMode | None
    temperature: int | None
    fan: FujitsuAcFanSpeed | None
    swing: FujitsuAcSwing | None
    power: bool

    def __init__(
        self,
        *,
        mode: FujitsuAcMode | None = None,
        temperature: int | None = None,
        fan: FujitsuAcFanSpeed = FujitsuAcFanSpeed.AUTO,
        swing: FujitsuAcSwing = FujitsuAcSwing.OFF,
        power: bool = True,
        modulation: int = 38000,
    ) -> None:
        """Initialize the Fujitsu A/C IR command."""
        super().__init__(modulation=modulation)

        if power:
            if mode is None:
                raise ValueError("mode is required while power is on")
            if temperature is None:
                raise ValueError("temperature is required while power is on")
            if not MIN_TEMP <= temperature <= MAX_TEMP:
                raise ValueError(
                    f"temperature {temperature} out of range {MIN_TEMP}..{MAX_TEMP}"
                )

        self.power = power
        # Powering off is a short frame that carries no state; storing fields it
        # cannot express would make a command unequal to itself after a
        # roundtrip.
        self.mode = mode if power else None
        self.temperature = temperature if power else None
        self.fan = fan if power else None
        self.swing = swing if power else None

    def _build_state(self) -> list[int]:
        """Build the state bytes, checksum included."""
        if not self.power:
            return [*_HEADER, _CMD_TURN_OFF, ~_CMD_TURN_OFF & 0xFF]

        assert (
            self.mode is not None
            and self.temperature is not None
            and self.fan is not None
            and self.swing is not None
        ), "state missing while power is on"

        temp = (self.temperature - _TEMP_OFFSET) * _TEMP_STEP
        state = [
            *_HEADER,
            _CMD_STATE,
            _REST_LENGTH,
            _PROTOCOL,
            _POWER_BIT | (temp << 2),
            self.mode,
            self.fan | (self.swing << 4),
            0x00,
            0x00,
            0x00,
            _BYTE14,
        ]
        state.append(_long_checksum(state))
        return state

    @override
    def get_raw_timings(self) -> list[int]:
        """Get raw timings for the Fujitsu A/C command."""
        timings: list[int] = [_HDR_MARK, -_HDR_SPACE]
        for byte in self._build_state():
            for i in range(8):
                timings.append(_BIT_MARK)
                timings.append(-(_ONE_SPACE if (byte >> i) & 1 else _ZERO_SPACE))
        timings.append(_BIT_MARK)
        return timings

    @classmethod
    def from_raw_timings(cls, timings: list[int]) -> Self | None:
        """Decode raw IR timings into a FujitsuAcCommand.

        Returns a FujitsuAcCommand if the timings match, or None otherwise.
        """
        # Header pair (2) + the shortest frame's bit pairs + the trailing mark
        if len(timings) < 2 + 2 * _SHORT_BITS + 1:
            return None
        if not _matches_header(timings[0], abs(timings[1])):
            return None

        bit_count = (len(timings) - 3) // 2
        if bit_count not in (_SHORT_BITS, _LONG_BITS):
            return None

        state: list[int] = []
        for byte_index in range(bit_count // 8):
            byte = 0
            for bit_index in range(8):
                i = 2 + 2 * (byte_index * 8 + bit_index)
                bit = _decode_bit(timings[i], abs(timings[i + 1]))
                if bit is None:
                    return None
                byte |= bit << bit_index
            state.append(byte)

        if not _is_close(timings[2 + 2 * bit_count], _BIT_MARK):
            return None
        if state[: len(_HEADER)] != _HEADER:
            return None

        if bit_count == _SHORT_BITS:
            return cls._from_short_state(state)
        return cls._from_long_state(state)

    @classmethod
    def _from_short_state(cls, state: list[int]) -> Self | None:
        """Decode a short frame, which only ever turns the unit off."""
        if state[-1] != ~state[-2] & 0xFF:
            return None
        # Other short frames exist, such as the econo and powerful toggles, but
        # they carry settings this class does not model.
        if state[_CMD_INDEX] != _CMD_TURN_OFF:
            return None
        return cls(power=False)

    @classmethod
    def _from_long_state(cls, state: list[int]) -> Self | None:
        """Decode a state frame."""
        if state[_CMD_INDEX] != _CMD_STATE:
            return None
        # Another variant would set a different version here, and its frame
        # would not mean what this one does.
        if state[_PROTOCOL_INDEX] != _PROTOCOL:
            return None
        if state[-1] != _long_checksum(state):
            return None
        if state[8] & _FAHRENHEIT_BIT:
            return None

        temp = (state[8] >> 2) & 0x3F
        if temp % _TEMP_STEP:
            return None
        temperature = temp // _TEMP_STEP + _TEMP_OFFSET
        if not MIN_TEMP <= temperature <= MAX_TEMP:
            return None

        try:
            mode = FujitsuAcMode(state[9] & 0b111)
            fan = FujitsuAcFanSpeed(state[10] & 0b111)
            swing = FujitsuAcSwing((state[10] >> 4) & 0b11)
        except ValueError:
            return None

        # Byte 8's power bit separates a frame that turns the unit on from one
        # that only changes a setting on a running unit. Either way the unit
        # ends up on, which is what this reports.
        return cls(
            mode=mode,
            temperature=temperature,
            fan=fan,
            swing=swing,
            power=True,
        )
