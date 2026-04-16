#!/bin/bash
# 2_batch_extract.sh (Multi-Profile Version)

PCAP_PATH=$(ls -t ./captures/*.pcap | head -1)
ANALYSIS_ROOT="./analysis"

if [ -z "$PCAP_PATH" ]; then echo "No PCAP found!"; exit 1; fi

SESSION_NAME=$(basename "$PCAP_PATH" .pcap)
SESSION_DIR="$ANALYSIS_ROOT/$SESSION_NAME"
mkdir -p "$SESSION_DIR"

echo "--- SESSION: $SESSION_NAME ---"

# Step 1: Find every unique (MAC, BW, Nc, Nr) tuple that has 5+ packets
tshark -r "$PCAP_PATH" -Y "wlan.vht.mimo_control.ncindex" -T fields \
-e wlan.sa -e wlan.vht.mimo_control.chanwidth -e wlan.vht.mimo_control.ncindex -e wlan.vht.mimo_control.nrindex \
| sort | uniq -c | sort -nr | while read -r count mac bw_hex nc_hex nr_hex; do
    
    if [ "$count" -lt 5 ]; then continue; fi

    # Convert Hex to Dec
    bw_dec=$((bw_hex)); nc_dec=$((nc_hex)); nr_dec=$((nr_hex))
    case $bw_dec in 0) bw=20 ;; 1) bw=40 ;; 2) bw=80 ;; *) bw=40 ;; esac
    nc=$((nc_dec + 1)); nr=$((nr_dec + 1))
    MIMO_STR="${nr}x${nc}"

    # Step 2: Create a Unique Folder Name
    # Format: results_MAC_MIMOCONFIG_BW
    MAC_CLEAN=${mac//:/}
    client_dir="$SESSION_DIR/results_${MAC_CLEAN}_${MIMO_STR}_${bw}MHz"
    mkdir -p "$client_dir"

    echo "------------------------------------------------"
    echo "Profile: $mac | $MIMO_STR @ ${bw}MHz | Packets: $count"

    # Step 3: Create a temporary filtered PCAP for just THIS profile
    # This ensures main.py doesn't get confused by other packets from the same MAC
    TEMP_PCAP="$client_dir/subset.pcap"
    tshark -r "$PCAP_PATH" -Y "wlan.sa == $mac and wlan.vht.mimo_control.chanwidth == $bw_hex and wlan.vht.mimo_control.ncindex == $nc_hex and wlan.vht.mimo_control.nrindex == $nr_hex" -w "$TEMP_PCAP"

    # Step 4: Extract RSSI (from the filtered subset)
    tshark -r "$TEMP_PCAP" -T fields -e radiotap.dbm_antsignal > "$client_dir/rssi.csv"

    # Step 5: Run main.py on the subset
    # Note: count is now exactly the number of packets in the subset
    LOWER_MAC=$(echo "$mac" | tr '[:upper:]' '[:lower:]')
    python main.py "$TEMP_PCAP" AC SU "$MIMO_STR" "$bw" "$LOWER_MAC" "$count" "$client_dir/v_matrix.npy" "$client_dir/angles.npy"

    # Cleanup temp file
    rm "$TEMP_PCAP"
done
