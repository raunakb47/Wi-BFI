"""
Module: capture_reader.py
Read 802.11 frames from a capture file: pcap or pcapng, either byte order and
either timestamp resolution.

Timestamp, addresses, signal and frame bytes all sit at fixed offsets in the
radiotap or 802.11 header, so frames are read directly rather than through a
dissector.

Frames are selected by Action Category, not subtype: reports arrive as Action
or Action No Ack depending on the device, so a subtype match misses whichever
is not listed. Category 0x15 is VHT and 0x1e is HE, which makes the standard a
property of the frame rather than a run-wide setting.
"""
import struct

LINKTYPE_IEEE802_11_RADIOTAP = 127

CATEGORY_VHT = 0x15
CATEGORY_HE = 0x1e
ACTION_COMPRESSED_BEAMFORMING = 0x00
STANDARD_FOR_CATEGORY = {CATEGORY_VHT: "AC", CATEGORY_HE: "AX"}

# Radiotap field alignment and size by "present" bit. Every defined bit needs
# an entry even when unwanted: fields are laid out in bit order, so a missing
# entry leaves the cursor short and later fields read at the wrong offset.
_RADIOTAP_FIELDS = {
    0:  (8, 8),    # TSFT
    1:  (1, 1),    # Flags
    2:  (1, 1),    # Rate
    3:  (2, 4),    # Channel
    4:  (1, 2),    # FHSS
    5:  (1, 1),    # Antenna signal (dBm)
    6:  (1, 1),    # Antenna noise (dBm)
    7:  (2, 2),    # Lock quality
    8:  (2, 2),    # TX attenuation
    9:  (2, 2),    # dB TX attenuation
    10: (1, 1),    # TX power (dBm)
    11: (1, 1),    # Antenna
    12: (1, 1),    # Antenna signal (dB)
    13: (1, 1),    # Antenna noise (dB)
    14: (2, 2),    # RX flags
    15: (2, 2),    # TX flags
    16: (1, 1),    # RTS retries
    17: (1, 1),    # Data retries
    18: (4, 8),    # XChannel
    19: (1, 3),    # MCS
    20: (4, 8),    # A-MPDU status
    21: (2, 12),   # VHT
    22: (8, 12),   # Timestamp
    23: (2, 12),   # HE
    24: (2, 12),   # HE-MU
    25: (2, 6),    # HE-MU-other-user
    26: (1, 1),    # 0-length PSDU
    27: (2, 4),    # L-SIG
}
# Bit 28 carries a variable-length TLV list, which a fixed field table cannot
# walk; parsing stops there.
_BIT_TLVS = 28
_BIT_DBM_ANTENNA_SIGNAL = 5
_BIT_FLAGS = 1
_BIT_CHANNEL = 3
_FLAG_BAD_FCS = 0x40
_BIT_RADIOTAP_NS_NEXT = 29
_BIT_EXT = 31


def radiotap_signal_dbm(buf):
    """
    Every dBm Antenna Signal value in the header, in stored order.

    A multi-chain adapter repeats the field per chain in successive radiotap
    namespaces; the first entry is the chain-agnostic value where present.
    Empty when the header carries no signal field.
    """
    if len(buf) < 8:
        return []
    header_len = struct.unpack_from('<H', buf, 2)[0]
    present_words = []
    offset = 4
    while offset + 4 <= header_len:
        word = struct.unpack_from('<I', buf, offset)[0]
        present_words.append(word)
        offset += 4
        if not word & (1 << _BIT_EXT):
            break

    values = []
    for word in present_words:
        if word & (1 << _BIT_TLVS):
            return values
        for bit in sorted(_RADIOTAP_FIELDS):
            if not word & (1 << bit):
                continue
            align, size = _RADIOTAP_FIELDS[bit]
            offset = (offset + align - 1) // align * align
            if offset + size > header_len:
                return values
            if bit == _BIT_DBM_ANTENNA_SIGNAL:
                values.append(struct.unpack_from('<b', buf, offset)[0])
            offset += size
        if not word & (1 << _BIT_RADIOTAP_NS_NEXT):
            break
    return values


def radiotap_channel_mhz(buf):
    """
    The Channel field's frequency in MHz, or None when the header omits it.

    Every bearing estimate depends on wavelength, which the capture frequency
    sets; 5180 and 5540 MHz differ by 7 percent. Read rather than configured.
    """
    if len(buf) < 8:
        return None
    header_len = struct.unpack_from('<H', buf, 2)[0]
    offset = 4
    words = []
    while offset + 4 <= header_len:
        word = struct.unpack_from('<I', buf, offset)[0]
        words.append(word)
        offset += 4
        if not word & (1 << _BIT_EXT):
            break
    if not words or not words[0] & (1 << _BIT_CHANNEL):
        return None
    word = words[0]
    # Fields preceding Channel, in bit order, honouring alignment.
    for bit in (0, 1, 2):
        if not word & (1 << bit):
            continue
        align, size = _RADIOTAP_FIELDS[bit]
        offset = (offset + align - 1) // align * align
        offset += size
    align, size = _RADIOTAP_FIELDS[_BIT_CHANNEL]
    offset = (offset + align - 1) // align * align
    if offset + size > header_len:
        return None
    return int(struct.unpack_from('<H', buf, offset)[0])


def radiotap_bad_fcs(buf):
    """
    True when the card found the frame's FCS wrong, False when it passed, None
    when the header carries no Flags field.

    Radiotap is written by the card and not covered by the 802.11 FCS, so
    readings taken from it stay valid on a failed frame. The failure
    invalidates the frame body, including the addresses identifying it.
    """
    if len(buf) < 8:
        return None
    header_len = struct.unpack_from('<H', buf, 2)[0]
    offset = 4
    words = []
    while offset + 4 <= header_len:
        word = struct.unpack_from('<I', buf, offset)[0]
        words.append(word)
        offset += 4
        if not word & (1 << _BIT_EXT):
            break
    if not words or not words[0] & (1 << _BIT_FLAGS):
        return None
    # Flags is the second defined field; only TSFT can precede it.
    if words[0] & (1 << 0):
        offset = (offset + 7) // 8 * 8
        offset += 8
    if offset >= header_len:
        return None
    return bool(buf[offset] & _FLAG_BAD_FCS)


def feedback_type_of(frame, standard):
    """
    "SU" or "MU" from the Feedback Type subfield of MIMO Control, or None for
    a value neither names (HE also defines a CQI-only report).

    Split out so a caller can ask which feedback types a capture holds without
    reconstructing anything, and so the bit positions live in one place.
    """
    control = frame[26:31] if standard == "AX" else frame[26:29]
    value = int.from_bytes(control, 'little')
    reported = (value >> 10) & 0x03 if standard == "AX" else (value >> 11) & 0x01
    return {0: "SU", 1: "MU"}.get(reported)


def _mac(raw):
    return ':'.join(f'{byte:02x}' for byte in raw)


def _pcap_records(handle):
    header = handle.read(24)
    if len(header) < 24:
        return
    magic = struct.unpack_from('<I', header, 0)[0]
    if magic in (0xa1b2c3d4, 0xa1b23c4d):
        endian, nanosecond = '<', magic == 0xa1b23c4d
    elif magic in (0xd4c3b2a1, 0x4d3cb2a1):
        endian, nanosecond = '>', magic == 0x4d3cb2a1
    else:
        raise ValueError('not a pcap file')
    if struct.unpack_from(endian + 'I', header, 20)[0] != LINKTYPE_IEEE802_11_RADIOTAP:
        raise ValueError('capture link type is not radiotap')
    divisor = 1e9 if nanosecond else 1e6
    while True:
        record = handle.read(16)
        if len(record) < 16:
            return
        seconds, fraction, captured, _ = struct.unpack(endian + 'IIII', record)
        payload = handle.read(captured)
        if len(payload) < captured:
            return
        yield seconds + fraction / divisor, payload


def _pcapng_records(handle):
    handle.seek(0)
    endian = '<'
    resolutions = {}
    interface = 0
    while True:
        block_header = handle.read(8)
        if len(block_header) < 8:
            return
        block_type = struct.unpack_from(endian + 'I', block_header, 0)[0]
        if block_type == 0x0A0D0D0A:                      # Section Header Block
            byte_order = struct.unpack_from('<I', block_header, 4)[0]
            # Length is byte-order dependent; re-read once known.
            endian = '<' if struct.unpack_from('<I', handle.read(4), 0)[0] == 0x1A2B3C4D else '>'
            handle.seek(-4, 1)
            block_length = struct.unpack_from(endian + 'I', block_header, 4)[0]
            resolutions, interface = {}, 0
        else:
            block_length = struct.unpack_from(endian + 'I', block_header, 4)[0]
        if block_length < 12:
            return
        body = handle.read(block_length - 12)
        if len(body) < block_length - 12:
            return
        handle.read(4)                                     # trailing block length

        if block_type == 0x00000001:                       # Interface Description Block
            link_type = struct.unpack_from(endian + 'H', body, 0)[0]
            resolution = 6
            offset = 8
            while offset + 4 <= len(body):
                code, length = struct.unpack_from(endian + 'HH', body, offset)
                offset += 4
                if code == 0:
                    break
                if code == 9 and length >= 1:              # if_tsresol
                    raw = body[offset]
                    resolution = raw & 0x7f if not raw & 0x80 else raw & 0x7f
                offset += (length + 3) // 4 * 4
            resolutions[interface] = (link_type, resolution)
            interface += 1
        elif block_type == 0x00000006:                     # Enhanced Packet Block
            iface, high, low, captured, _ = struct.unpack_from(endian + 'IIIII', body, 0)
            link_type, resolution = resolutions.get(iface, (LINKTYPE_IEEE802_11_RADIOTAP, 6))
            if link_type != LINKTYPE_IEEE802_11_RADIOTAP:
                continue
            timestamp = ((high << 32) | low) / (10 ** resolution)
            yield timestamp, body[20:20 + captured]


def _records(path):
    with open(path, 'rb') as handle:
        magic = handle.read(4)
        handle.seek(0)
        if magic == b'\x0a\x0d\x0d\x0a':
            yield from _pcapng_records(handle)
        else:
            yield from _pcap_records(handle)


def beamforming_reports(path, feedback_type=None):
    """
    Yield one dict per compressed beamforming report.

    feedback_type selects "SU" or "MU"; None yields both. Only that subfield is
    read here, the caller decodes the rest.
    """
    wanted = feedback_type if feedback_type in ("SU", "MU") else None
    for timestamp, buf in _records(path):
        if len(buf) < 8:
            continue
        header_len = struct.unpack_from('<H', buf, 2)[0]
        frame = buf[header_len:]
        if len(frame) < 32:
            continue
        frame_control = frame[0]
        if frame_control & 0x0c:                           # management frames only
            continue
        if frame_control >> 4 not in (13, 14):             # Action, Action No Ack
            continue
        category = frame[24]
        if category not in STANDARD_FOR_CATEGORY or frame[25] != ACTION_COMPRESSED_BEAMFORMING:
            continue

        standard = STANDARD_FOR_CATEGORY[category]
        reported = feedback_type_of(frame, standard)
        if wanted is not None and reported != wanted:
            continue

        yield {
            "timestamp": timestamp,
            "standard": standard,
            "receiver": _mac(frame[4:10]),
            "transmitter": _mac(frame[10:16]),
            "signal_dbm": radiotap_signal_dbm(buf),
            "bad_fcs": radiotap_bad_fcs(buf),
            "freq_mhz": radiotap_channel_mhz(buf),
            "feedback": reported,
            "raw": buf.hex(),
            "radiotap_len": header_len,
        }
