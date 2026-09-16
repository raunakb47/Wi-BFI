#!/bin/bash
# 1_capture.sh

DURATION="15m"
INTERFACE="wlan0"
SAVE_DIR="../bfi-workspace/captures"

# Create the captures directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Generate a filename like: bfi_trace_20260402_1130.pcap
FILENAME="$SAVE_DIR/bfi_trace_$(date +%Y%m%d_%H%M).pcap"

echo "--- Starting $DURATION BFI Capture on $INTERFACE ---"
echo "Saving raw trace to: $FILENAME"

# Compressed beamforming reports only. wlan[24] is the Action frame's Category
# and wlan[25] its Action code, at the same offset in Action and Action No Ack;
# category 0x15 is VHT and 0x1e is HE, action 0x00 is Compressed Beamforming.
# Both categories are matched because the extractor reads the standard from this
# byte and decodes Wi-Fi 5 and Wi-Fi 6 feedback from one capture.
sudo timeout $DURATION tcpdump -i $INTERFACE -s 0 -n \
    "wlan[24:2] == 0x1500 or wlan[24:2] == 0x1e00" -w "$FILENAME"

echo "--- Capture Complete. File saved in $SAVE_DIR. Ready for Step 2 (Batch Extraction). ---"
