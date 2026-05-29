"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Import necessary libraries
import pyshark
import numpy as np
import math
from textwrap import wrap
import argparse
from vmatrices import vmatrices
from bfi_angles import bfi_angles
from utils import hex2dec, flip_hex

# Set the default value for the least significant bit (LSB)
LSB = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IEEE 802.11 Agnostic BFI Extraction Engine")

    # Define command-line arguments
    parser.add_argument('file_name', help='File name to process (PCAP)')
    parser.add_argument('standard', help='Operating standard: "AC" or "AX"')
    parser.add_argument('mimo', help='Network formation: "SU" (Single User) or "MU" (Multi User)')
    parser.add_argument('config', help='Fallback antenna config (e.g., 4x4, 4x2, 2x2)')
    parser.add_argument('bw', help='Bandwidth of the capture (20, 40, 80, 160)')
    parser.add_argument('num_packet_to_process', help='Maximum packets to process')
    parser.add_argument('saved_vmatrices', help='Output numpy file for V-Matrices')
    parser.add_argument('saved_angles', help='Output numpy file for Raw Angles')

    args = parser.parse_args()

    file_name = args.file_name
    standard = args.standard
    mimo = args.mimo
    fallback_config = args.config
    bw = int(args.bw)
    num_packet_to_process = int(args.num_packet_to_process)
    saved_vmatrices = args.saved_vmatrices
    saved_angles = args.saved_angles

    if mimo == "MU" and standard == "AX":
        print("[!] MU-MIMO is not available for AX yet. Feature pending.")
    else:
        print(f"[*] Processing {file_name} (Standard: {standard}, BW: {bw}MHz)")

    # ---------------------------------------------------------
    # Subcarrier Mapping
    # ---------------------------------------------------------
    if standard == "AC":
        if bw == 80:
            subcarrier_idxs = np.arange(-122, 123)
            pilot_n_null = np.array([-104, -76, -40, -12, -1, 0, 1, 10, 38, 74, 102])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 40:
            subcarrier_idxs = np.arange(-58, 59)
            pilot_n_null = np.array([-54, -26, -12, -1, 0, 1, 10, 24, 52])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 20:
            subcarrier_idxs = np.arange(-28, 29)
            pilot_n_null = np.array([-21, -8, 0, 6, 21])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)

    if standard == "AX":
        if bw == 160:
            subcarrier_idxs = np.arange(-1012, 1013, 4)
            pilot_n_null = np.array([-512, -8, -4, 0, 4, 8, 512])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 80:
            subcarrier_idxs = np.arange(-500, 504, 4)
            pilot_n_null = np.array([0])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 40:
            subcarrier_idxs = np.arange(-244, 248, 4)
            pilot_n_null = np.array([0])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 20:
            neg_subcarriers = np.setdiff1d(np.arange(-122, 0, 2), np.arange(-118, -2, 4))
            pos_subcarriers = np.setdiff1d(np.arange(2, 124, 2), np.arange(6, 122, 4))
            subcarrier_idxs = np.concatenate((neg_subcarriers, pos_subcarriers))

    # ---------------------------------------------------------
    # Agnostic Packet Filtering (No MAC defined)
    # ---------------------------------------------------------
    if standard == "AX":
        display_filter = f'wlan.he.mimo.feedback_type=={mimo}'
    else:
        display_filter = f'wlan.vht.mimo_control.feedbacktype=={mimo}'

    packets = pyshark.FileCapture(
        input_file=file_name,
        display_filter=display_filter,
        use_json=True,
        include_raw=True
    )._packets_from_tshark_sync()

    buckets_v_matrices = {}
    buckets_angles = {}

    for p in range(num_packet_to_process):
        try:
            current_packet = packets.__next__()
        except StopIteration:
            break
            
        packet_raw = current_packet.frame_raw.value

        try:
            mac_addr = current_packet.wlan.ta
            # Critical Requirement for Temporal Sanitization
            timestamp = float(current_packet.sniff_timestamp) 
        except AttributeError:
            continue 

        try:
            if standard == "AX":
                nc_idx = int(current_packet.wlan.he_mimo_control_ncidx)
                nr_idx = int(current_packet.wlan.he_mimo_control_nridx)
            else:
                nc_idx = int(current_packet.wlan.vht_mimo_control_ncindex)
                nr_idx = int(current_packet.wlan.vht_mimo_control_nridx)
            pkt_config = f"{nr_idx + 1}x{nc_idx + 1}"
        except AttributeError:
            pkt_config = fallback_config

        bucket_key = f"{mac_addr}_{pkt_config}"
        
        if bucket_key not in buckets_v_matrices:
            buckets_v_matrices[bucket_key] = []
            buckets_angles[bucket_key] = []

        # ---------------------------------------------------------
        # Hex Header Traversal
        # ---------------------------------------------------------
        Header_length_dec = hex2dec(flip_hex(packet_raw[4:8]))
        i = Header_length_dec * 2

        if standard == "AX":
            packet_mimo_control = packet_raw[(i + 52):(i + 62)]
            packet_mimo_control_binary = ''.join(format(int(char, 16), '04b') for char in flip_hex(packet_mimo_control))
            codebook_info = packet_mimo_control_binary[30] 
            packet_snr = packet_raw[(i + 62):(i + 62 + 2*int(pkt_config[-1]))]

        if standard == "AC":
            packet_mimo_control = packet_raw[(i + 52):(i + 58)]
            packet_mimo_control_binary = ''.join(format(int(char, 16), '04b') for char in flip_hex(packet_mimo_control))
            codebook_info = packet_mimo_control_binary[13]
            packet_snr = packet_raw[(i + 58):(i + 58 + 2*int(pkt_config[-1]))]

        if mimo == "SU":
            if codebook_info == "1":
                psi_bit = 4
            else:
                psi_bit = 2
            phi_bit = psi_bit + 2
        elif mimo == "MU":
            if codebook_info == "1":
                psi_bit = 7
            else:
                psi_bit = 5
            phi_bit = psi_bit + 2

        # ---------------------------------------------------------
        # Definitions
        # ---------------------------------------------------------
        if pkt_config == "4x4" or pkt_config == "4x3":
            Nc_users = int(pkt_config[-1])
            Nr = 4 
            phi_numbers = 6
            psi_numbers = 6
            order_angles = ['phi_11', 'phi_21', 'phi_31', 'psi_21', 'psi_31', 'psi_41', 
                            'phi_22', 'phi_32', 'psi_32', 'psi_42', 'phi_33', 'psi_43']
            order_bits = [phi_bit]*3 + [psi_bit]*3 + [phi_bit]*2 + [psi_bit]*2 + [phi_bit]*1 + [psi_bit]*1
            tot_angles_users = phi_numbers + psi_numbers
            tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit

        elif pkt_config == "4x2":
            Nc_users = 2 
            Nr = 4 
            phi_numbers = 5
            psi_numbers = 5
            order_angles = ['phi_11', 'phi_21', 'phi_31', 'psi_21', 'psi_31', 'psi_41', 
                            'phi_22', 'phi_32', 'psi_32', 'psi_42']
            order_bits = [phi_bit, phi_bit, phi_bit, psi_bit, psi_bit, psi_bit, phi_bit, phi_bit, psi_bit, psi_bit]
            tot_angles_users = phi_numbers + psi_numbers
            tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit

        elif pkt_config == "4x1":
            Nc_users = 1 
            Nr = 4 
            phi_numbers = 3
            psi_numbers = 3
            order_angles = ['phi_11', 'phi_21', 'phi_31', 'psi_21', 'psi_31', 'psi_41']
            order_bits = [phi_bit, phi_bit, phi_bit, psi_bit, psi_bit, psi_bit]
            tot_angles_users = phi_numbers + psi_numbers
            tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit

        elif pkt_config == "3x3" or pkt_config == "3x2":
            Nc_users = int(pkt_config[-1]) 
            Nr = 3 
            phi_numbers = 3
            psi_numbers = 3
            order_angles = ['phi_11', 'phi_21', 'psi_21', 'psi_31', 'phi_22', 'psi_32']
            order_bits = [phi_bit, phi_bit, psi_bit, psi_bit, phi_bit, psi_bit]
            tot_angles_users = phi_numbers + psi_numbers
            tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit

        elif pkt_config == "3x1":
            Nc_users = 1 
            Nr = 3 
            phi_numbers = 2
            psi_numbers = 2
            order_angles = ['phi_11', 'phi_21', 'psi_21', 'psi_31']
            order_bits = [phi_bit, phi_bit, psi_bit, psi_bit]
            tot_angles_users = phi_numbers + psi_numbers
            tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit

        elif pkt_config == "2x2" or pkt_config == "2x1":
            Nc_users = int(pkt_config[-1]) 
            Nr = 2 
            phi_numbers = 1
            psi_numbers = 1
            order_angles = ['phi_11', 'psi_21']
            order_bits = [phi_bit, psi_bit]
            tot_angles_users = phi_numbers + psi_numbers
            tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit

        else:
            continue

        NSUBC_VALID = len(subcarrier_idxs)
        
        # ---------------------------------------------------------
        # BFI Payload Extraction
        # ---------------------------------------------------------
        if standard == "AX":
            Feedback_angles = packet_raw[(i + 62 + 2*int(pkt_config[-1])):(len(packet_raw) - 8)]
        if standard == "AC":
            Feedback_angles = packet_raw[(i + 58 + 2*int(pkt_config[-1])):(len(packet_raw) - 8)]
            
        Feedback_angles_splitted = np.array(wrap(Feedback_angles, 2))
        Feedback_angles_bin = ""

        for idx in range(0, len(Feedback_angles_splitted)):
            bin_str = str(format(hex2dec(Feedback_angles_splitted[idx]), '08b'))
            if LSB:
                bin_str = bin_str[::-1]
            Feedback_angles_bin += bin_str

        Feed_back_angles_bin_chunk = np.array(wrap(Feedback_angles_bin[:(tot_bits_users * NSUBC_VALID)], tot_bits_users))

        angle = bfi_angles(Feed_back_angles_bin_chunk, LSB, NSUBC_VALID, order_bits)
        
        # Reconstruct the complex Matrix
        v_matrix = vmatrices(angle, phi_bit, psi_bit, NSUBC_VALID, Nr, Nc_users, pkt_config)
        
        # Merge the absolute timestamp with the Spatial Matrix for VSS-LMS interpolation
        buckets_v_matrices[bucket_key].append((timestamp, v_matrix))
        
        # Merge the absolute timestamp with the raw angles for external debug/logging
        buckets_angles[bucket_key].append((timestamp, angle))

    np.save(saved_vmatrices, buckets_v_matrices)
    np.save(saved_angles, buckets_angles)
    print(f"[*] Extraction complete. Saved to {saved_vmatrices}")