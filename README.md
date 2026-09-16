# Modified Wi-BFI

Fork of [Wi-BFI: Extracting the IEEE 802.11 Beamforming Feedback Information from
Commercial Wi-Fi Devices](https://dl.acm.org/doi/10.1145/3615453.3616514) by
Foysal Haque et al. Upstream: <https://github.com/kfoysalhaque/Wi-BFI>.

Wi-BFI is the first open-source tool for retrieving Wi-Fi beamforming feedback
angles (BFAs) and reconstructing beamforming feedback information (BFI) in the
form of V-matrices. This fork keeps that capability and the same command-line
signature, and changes how each frame is read so that one capture can be
processed without being told what is in it.

## What this fork changes

Upstream takes the standard, the antenna configuration and the channel width as
command-line arguments, applying them to every frame in the file. This fork
decodes all three from each frame and groups the results accordingly.

| | upstream | this fork |
|---|---|---|
| standard (11ac / 11ax) | command-line argument | read from the Action Category of each frame |
| antenna configuration | command-line argument | read from MIMO Control of each frame |
| channel width | command-line argument | read from MIMO Control of each frame |
| frame selection | `tshark` display filter via pyshark | capture file read directly |
| output | one stack of V-matrices | one stack per transmitter, receiver, configuration and width |

Consequences for the user:

- **A capture does not have to be homogeneous.** Several clients, several
  antenna configurations, several channel widths and both standards can sit in
  one file and all decode in one pass. Upstream requires one run per
  configuration, over a capture filtered down to it.
- **Wrong metadata cannot corrupt the output.** Antenna configuration selects the
  Givens codebook and channel width sets the subcarrier count, so a mismatch
  produces wrong V-matrices rather than a visible error.
- **Wireshark is no longer needed at runtime.** `tshark` remains useful for
  verifying a decode, but the extractor does not call it.

Additional changes, each against a case observed on real captures:

- Frames are selected on the Action Category rather than the frame subtype.
  Beamforming reports are sent as Action **or** Action No Ack depending on the
  device, and a subtype-13-only match misses every device that uses subtype 14.
- The fixed four-byte FCS trim is removed. The payload is
  prefix-sliced to the length the configuration requires, so trailing bytes are
  ignored whether or not an FCS is present.
- MU feedback is accepted on 802.11ax, not only on 802.11ac.
- The signal value is taken from the first radiotap antenna-signal field rather
  than the last. On a multi-chain adapter the last entry is one chain, several dB
  below the figure for the frame as received.
- A frame carrying fewer angle bits than its configuration requires is reported
  with the exact shortfall and skipped, instead of ending the run.
- A missing signal field is recorded as unmeasured rather than as a stand-in
  number.

V-matrices are bit-identical to upstream across 11ac SU, 11ac MU and 11ax SU at
20, 40, 80 and 160 MHz.

## Install

```
git clone https://github.com/raunakb47/Wi-BFI.git
cd Wi-BFI
conda env create -f wi-bfi.yml
```

`tshark`, `wireshark` and `aircrack-ng` are useful for capturing and for
cross-checking a decode, but the extractor does not require them:

```
sudo apt-get install tshark wireshark aircrack-ng
```

## Usage

```
python main.py <file> <standard> <mimo> <config> <bandwidth> <packets> <vmatrix_out> <bfa_out>
```

Example:

```
python main.py traces/11ax_SU_4x2_160.pcapng AX SU 4x2 160 200 V_out.npy bfa_out.npy
```

| argument | meaning |
|---|---|
| `file` | pcap or pcapng trace |
| `standard` | **ignored**; read per frame. Kept so the argument list matches upstream |
| `mimo` | `SU` or `MU`. Sets the angle quantisation and selects which frames are read |
| `config` | **ignored**; read per frame |
| `bandwidth` | **ignored**; read per frame |
| `packets` | maximum number of frames to process |
| `vmatrix_out` | output `.npy` for the V-matrices |
| `bfa_out` | output `.npy` for the raw angles |

Upstream's ninth argument, the beamformee MAC address, is gone: every
transmitter in the capture is decoded and the address becomes part of the output
key instead of a filter. `standard`, `config` and `bandwidth` are accepted and
ignored so that existing command lines keep working.

## Output

Both files hold a dictionary keyed by

```
"{transmitter}_{receiver}_{Nr}x{Nc}@{bandwidth}"
e.g. "f6:b1:4f:a6:7b:7c_78:0c:f0:7a:5e:6e_4x2@40"
```

Load with `numpy.load(path, allow_pickle=True).item()`. Each key maps to a list
of per-frame tuples:

```python
vmatrix_out : (timestamp, v_matrix, rssi, stream_snr, signal_chains)
bfa_out     : (timestamp, angles)
```

| field | description |
|---|---|
| `timestamp` | epoch seconds, from the capture |
| `v_matrix` | complex array `(subcarriers, Nr, Nc)`, columns orthonormal |
| `rssi` | dBm at the monitor, or `None` if the header carried no signal field |
| `stream_snr` | average SNR in dB per space-time stream, as reported by the beamformee |
| `signal_chains` | every radiotap antenna-signal value, combined figure first |
| `angles` | raw quantised Givens angles, `(subcarriers, n_angles)` |

`rssi` and `stream_snr` describe **different links**. `rssi` is what the monitor
received from the beamformee. `stream_snr` is the beamformee's own measurement of
the link it was sounded on, so it describes the beamformer-to-beamformee path
and moves independently of the monitor's reading.

`signal_chains[0]` is the same number as `rssi`. The remaining entries are the
per-chain values. What the first value means is a property of the driver — on one
adapter it is the chains summed, on another the strongest chain — so the whole
list is carried and the distinction can be made downstream.

## Helper scripts

| script | purpose |
|---|---|
| `1_capture.sh` | monitor-mode capture of VHT and HE beamforming reports into a rotating file |
| `2_batch_extract.sh` | run the extractor over the newest capture, once for SU and once for MU, then report every bucket found with its frame count, array shape, duration, RSSI range and SNR range |
| `3_visualize.py` | one figure per bucket: monitor RSSI, reported SNR per stream, transmit-antenna power share, and that share per subcarrier |

`2_batch_extract.sh` doubles as a check that a capture decoded correctly: each
bucket's array shape is verified against the key naming it, and the subcarrier
count identifies the standard independently of what the frame claimed.

## `capture_reader.py`

Reads pcap and pcapng directly and yields one record per beamforming report.
Everything the extractor consumes — timestamp, addresses, signal, frame bytes —
sits at a fixed offset in the radiotap or 802.11 header, so no dissector is
involved. It handles either byte order and either timestamp resolution, and walks
the radiotap present-bit chain to recover every antenna-signal field.

It also fixes a capture-side point. A BPF filter such as

```
wlan[24:2] == 0x1500 or wlan[24:2] == 0x1e00
```

tests bytes 24 and 25 of the frame, which are the Action Category and Action code
only in a management Action frame. In a data frame those bytes are the start of
the frame body, so an occasional data frame passes the filter by coincidence.
`capture_reader` checks the frame type and subtype before the category, so such
frames are discarded rather than decoded.

## Original tooling

`main_live_plot.py`, `plot.py`, `plot_spectrogram.py`, `plot_vmatrices.py`,
`main_extract_batch.py` and the `Demo` directory are upstream's and are
unchanged. The live-plot workflow is documented in the
[upstream README](https://github.com/kfoysalhaque/Wi-BFI) with a
[video demonstration](https://youtu.be/0k7uYRCmMBw?si=HGkOYjEC4bzo8V7U).

## Limitations

- 802.11ax MU has not been verified end to end; no such capture was available.
  If the subcarrier set is wrong for a given MU report the short-payload guard
  reports the shortfall rather than decoding past it.
- Coarser subcarrier groupings, HE feedback scoped to a narrow resource unit, and
  VHT at 160 MHz are not implemented. Diagnostic present.
  than guessed.

## Credit

The tool, the paper and the original implementation are the work of
[Foysal Haque](https://kfoysalhaque.github.io/) and co-authors. For questions
about the original project, contact **haque.k@northeastern.edu**. This fork's
changes are limited to the extraction path described above.
