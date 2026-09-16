#!/bin/bash
# 2_batch_extract.sh

ANALYSIS_ROOT="../bfi-workspace/analysis"
MAX_PACKETS="${MAX_PACKETS:-1000000}"

PCAP_PATH="${1:-$(ls -t ../bfi-workspace/captures/*.pcap ../bfi-workspace/captures/*.pcapng 2>/dev/null | head -1)}"
if [ -z "$PCAP_PATH" ]; then echo "No capture found in ../bfi-workspace/captures"; exit 1; fi

SESSION_NAME=$(basename "$PCAP_PATH"); SESSION_NAME="${SESSION_NAME%.*}"
SESSION_DIR="$ANALYSIS_ROOT/$SESSION_NAME"
mkdir -p "$SESSION_DIR"

echo "--- SESSION: $SESSION_NAME ---"
echo "Capture: $PCAP_PATH"

# main.py groups reports by transmitter, receiver, antenna configuration and
# channel width on its own, and reads the standard from each frame's Action
# Category, so one pass covers every VHT and HE talker in the capture. Splitting
# the capture into one subset per configuration is no longer needed.
#
# Feedback type is the one setting still chosen per run: SU and MU quantise the
# Givens angles to different bit widths, so a run reads only the frames matching
# the type it was given. The standard, configuration and channel width arguments
# are placeholders kept for the positional CLI signature; each is decoded per
# packet and the values passed here are ignored.
for FEEDBACK in SU MU; do
    OUT_DIR="$SESSION_DIR/$FEEDBACK"
    mkdir -p "$OUT_DIR"
    echo
    echo "=== $FEEDBACK feedback ==="
    python3 main.py "$PCAP_PATH" AC "$FEEDBACK" 4x2 80 "$MAX_PACKETS" \
        "$OUT_DIR/v_matrix.npy" "$OUT_DIR/angles.npy"
done

# Bucket report. Each bucket's V-matrix stack is checked against its own key:
# the array shape must carry the (Nr, Nc) the key names, and the subcarrier
# count must be one the key's channel width defines. A mismatch means the MIMO
# Control decode and the reconstruction disagree, which is what this step is
# for; the standard column is read back from whichever subcarrier set matched.
echo
echo "=== BUCKETS ==="
python3 - "$SESSION_DIR" <<'PY'
import sys
from pathlib import Path

import numpy as np

from main import subcarrier_indices

session = Path(sys.argv[1])

for feedback in ("SU", "MU"):
    path = session / feedback / "v_matrix.npy"
    if not path.exists():
        continue
    buckets = np.load(path, allow_pickle=True).item()
    if not buckets:
        print(f"{feedback}: no reports")
        continue
    print(f"{feedback}:")
    for key, samples in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        transmitter, receiver, shape = key.split("_")
        config, bw = shape.split("@")
        nr, nc = (int(x) for x in config.split("x"))

        shapes = {s[1].shape for s in samples}
        if len(shapes) > 1:
            print(f"  {transmitter} -> {receiver}  {config} @ {bw} MHz")
            print(f"    reports {len(samples):6d}  MISMATCH ragged stack: {sorted(shapes)}")
            continue

        times = np.array([s[0] for s in samples])
        span = times.max() - times.min()
        stack = np.stack([s[1] for s in samples])
        rssi = np.array([np.nan if s[2] is None else s[2] for s in samples])
        snr = np.array([np.mean(s[3]) if s[3] else np.nan for s in samples])

        widths = {std: subcarrier_indices(std, int(bw)) for std in ("AC", "AX")}
        standard = next((std for std, idxs in widths.items()
                         if idxs is not None and len(idxs) == stack.shape[1]), "?")
        expected = (len(samples), stack.shape[1], nr, nc)
        verdict = "ok" if stack.shape == expected and standard != "?" else "MISMATCH"

        print(f"  {transmitter} -> {receiver}  {config} @ {bw} MHz  {standard}")
        print(f"    reports {len(samples):6d} over {span:8.1f} s "
              f"({len(samples) / span if span else float('nan'):.2f}/s)  V{stack.shape} {verdict}")
        print(f"    monitor RSSI {np.nanmin(rssi):6.1f} .. {np.nanmax(rssi):6.1f} dBm "
              f"({np.count_nonzero(np.isnan(rssi))} unmeasured)   "
              f"report SNR {np.nanmin(snr):5.2f} .. {np.nanmax(snr):5.2f} dB")
PY
