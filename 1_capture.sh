#!/bin/bash
# 1_capture.sh

DURATION="15m"
INTERFACE="wlan0"
SAVE_DIR="./captures"

# Create the captures directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Generate a filename like: bfi_trace_20260402_1130.pcap
FILENAME="$SAVE_DIR/bfi_trace_$(date +%Y%m%d_%H%M).pcap"

echo "--- Starting $DURATION BFI Capture on $INTERFACE ---"
echo "Saving raw trace to: $FILENAME"

# Capture only VHT Beamforming Reports
sudo timeout $DURATION tcpdump -i $INTERFACE -s 0 -n "wlan[24:2] == 0x1500" -w "$FILENAME"

echo "--- Capture Complete. File saved in $SAVE_DIR. Ready for Step 2 (Batch Extraction). ---"
