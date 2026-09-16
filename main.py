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

import sys

import numpy as np
import math
from textwrap import wrap
import argparse
from vmatrices import vmatrices
from bfi_angles import bfi_angles
from utils import hex2dec, flip_hex
from capture_reader import beamforming_reports

# Set the default value for the least significant bit (LSB)
LSB = True

# Channel Width subfield of the MIMO Control field, B6-B7 in both standards.
BW_FROM_INDEX = {0: 20, 1: 40, 2: 80, 3: 160}


def subcarrier_indices(standard, bw):
    """
    Subcarrier indices carrying a compressed beamforming feedback matrix, for one
    channel width. Returns None for a width the standard does not define.

    The 11ac sets are ungrouped (Ng=1); the 11ax sets step by 4 because HE
    feedback is always grouped, and Ng=4 is its finest setting.
    """
    if standard == "AC":
        if bw == 80:
            return np.setdiff1d(np.arange(-122, 123),
                                np.array([-104, -76, -40, -12, -1, 0, 1, 10, 38, 74, 102]))
        if bw == 40:
            return np.setdiff1d(np.arange(-58, 59),
                                np.array([-54, -26, -12, -1, 0, 1, 10, 24, 52]))
        if bw == 20:
            return np.setdiff1d(np.arange(-28, 29), np.array([-21, -8, 0, 6, 21]))
        return None

    if standard == "AX":
        if bw == 160:
            return np.setdiff1d(np.arange(-1012, 1013, 4),
                                np.array([-512, -8, -4, 0, 4, 8, 512]))
        if bw == 80:
            return np.setdiff1d(np.arange(-500, 504, 4), np.array([0]))
        if bw == 40:
            return np.setdiff1d(np.arange(-244, 248, 4), np.array([0]))
        if bw == 20:
            neg = np.setdiff1d(np.arange(-122, 0, 2), np.arange(-118, -2, 4))
            pos = np.setdiff1d(np.arange(2, 124, 2), np.arange(6, 122, 4))
            return np.concatenate((neg, pos))
        return None

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IEEE 802.11 Agnostic BFI Extraction Engine")
    parser.add_argument('file_name', help='File name to process (PCAP)')
    parser.add_argument('standard', help='Unused; the standard is decoded per packet from the Action Category. Kept for CLI compatibility.')
    parser.add_argument('mimo', help='Network formation: "SU" or "MU"')
    parser.add_argument('config', help='Unused; antenna config is decoded per packet from MIMO Control. Kept for CLI compatibility.')
    parser.add_argument('bw', help='Unused; channel width is decoded per packet from MIMO Control. Kept for CLI compatibility.')
    parser.add_argument('num_packet_to_process', help='Maximum packets to process')
    parser.add_argument('saved_vmatrices', help='Output numpy file for V-Matrices')
    parser.add_argument('saved_angles', help='Output numpy file for Raw Angles')

    args = parser.parse_args()

    file_name = args.file_name
    # Retained so the positional CLI signature stays valid for callers.
    _unused_standard_arg = args.standard
    mimo = args.mimo
    # Retained so the positional CLI signature stays valid for callers.
    _unused_config_arg = args.config
    # Retained so the positional CLI signature stays valid for callers.
    _unused_bw_arg = args.bw
    num_packet_to_process = int(args.num_packet_to_process)
    saved_vmatrices = args.saved_vmatrices
    saved_angles = args.saved_angles

    print(f"[*] Processing {file_name} ({mimo} feedback; standard read per packet)")

    # capture_reader selects frames by Action Category, which also identifies the
    # standard, so one capture may hold both VHT and HE feedback and both decode
    # in the same pass.
    buckets_v_matrices = {}
    buckets_angles = {}

    for p, record in enumerate(beamforming_reports(file_name, mimo)):
        if p >= num_packet_to_process:
            break

        packet_raw = record["raw"]
        standard = record["standard"]
        mac_addr_ta = record["transmitter"]
        mac_addr_ra = record["receiver"]
        timestamp = record["timestamp"]

        # A multi-chain adapter reports a chain-agnostic signal value first, then
        # one per chain; the first is the figure for the frame as received. A
        # header carrying no signal field yields None, which consumers see as NaN:
        # a stand-in value would be read downstream as measured received power.
        # The V-matrix does not depend on this field, so the packet is still kept.
        #
        # The whole list is carried alongside because what the first value means
        # is a property of the driver, not of the standard: on the adapter behind
        # the bundled 11ac traces it is the stronger chain, on an mt7921au it is
        # the two chains summed, and the two differ by up to 3 dB in a way that
        # moves with the chain balance. Only the per-chain values tell them apart.
        signal_chains = tuple(float(value) for value in record["signal_dbm"])
        rssi = signal_chains[0] if signal_chains else None

        # ---------------------------
        # Hex Header Traversal
        # ---------------------------
        Header_length_dec = hex2dec(flip_hex(packet_raw[4:8]))
        i = Header_length_dec * 2

        # Nc/Nr, channel width and codebook come from the MIMO Control field.
        # pkt_config selects the Givens codebook in vmatrices() and the channel
        # width sets the subcarrier count, so a wrong value corrupts the
        # V-matrices rather than only mislabelling the bucket; both are read per
        # packet, since one capture may carry several configurations.
        #
        # flip_hex reverses byte order, so binary index k holds spec bit
        # B(width-1-k): Nc Index is B0-B2, Nr B3-B5 and channel width B6-B7, in
        # both the VHT (3-byte) and HE (5-byte) fields.
        if standard == "AX":
            packet_mimo_control = packet_raw[(i + 52):(i + 62)]
            packet_mimo_control_binary = ''.join(format(int(char, 16), '04b') for char in flip_hex(packet_mimo_control))
            codebook_info = packet_mimo_control_binary[30]
            nc_idx = int(packet_mimo_control_binary[37:40], 2)
            nr_idx = int(packet_mimo_control_binary[34:37], 2)
            bw_idx = int(packet_mimo_control_binary[32:34], 2)

        if standard == "AC":
            packet_mimo_control = packet_raw[(i + 52):(i + 58)]
            packet_mimo_control_binary = ''.join(format(int(char, 16), '04b') for char in flip_hex(packet_mimo_control))
            codebook_info = packet_mimo_control_binary[13]
            nc_idx = int(packet_mimo_control_binary[21:24], 2)
            nr_idx = int(packet_mimo_control_binary[18:21], 2)
            bw_idx = int(packet_mimo_control_binary[16:18], 2)

        pkt_config = f"{nr_idx + 1}x{nc_idx + 1}"

        pkt_bw = BW_FROM_INDEX[bw_idx]
        subcarrier_idxs = subcarrier_indices(standard, pkt_bw)
        if subcarrier_idxs is None:
            print(f"[!] skipping packet {p}: {pkt_bw} MHz is not defined for {standard}", file=sys.stderr)
            continue

        # A bucket holds one stack of V-matrices of shape (NSUBC_VALID, Nr, Nc),
        # so channel width belongs in the key or the stack is ragged. '@' keeps
        # the key at three '_'-separated fields with "{Nr}x{Nc}" still parseable.
        bucket_key = f"{mac_addr_ta}_{mac_addr_ra}_{pkt_config}@{pkt_bw}"

        if bucket_key not in buckets_v_matrices:
            buckets_v_matrices[bucket_key] = []
            buckets_angles[bucket_key] = []

        # Average SNR of each space-time stream, one signed byte per stream at
        # the head of the Compressed Beamforming Report. This is the reporting
        # station's own measurement of the link it received the sounding on,
        # a different quantity from the monitor's radiotap signal above, and
        # the only figure in the frame describing the beamformer-to-beamformee
        # path. dB = 22 + 0.25 * value, over -10 to 53.75 dB.
        if standard == "AX":
            packet_snr = packet_raw[(i + 62):(i + 62 + 2*(nc_idx + 1))]
        if standard == "AC":
            packet_snr = packet_raw[(i + 58):(i + 58 + 2*(nc_idx + 1))]

        stream_snr = []
        for b in range(0, len(packet_snr) - 1, 2):
            value = hex2dec(packet_snr[b:b + 2])
            stream_snr.append(22 + 0.25 * (value - 256 if value > 127 else value))
        stream_snr = tuple(stream_snr)

        # Givens angle quantisation, from the Codebook Information subfield. The
        # SU and MU pairs are the same in VHT and HE: SU gives (psi, phi) of
        # (2, 4) or (4, 6), MU gives (5, 7) or (7, 9).
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

        # -------------------------
        # Definitions
        # -------------------------

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

        # ----------------------------
        # BFI Payload Extraction
        # ----------------------------
        # Read to the end of the frame. The payload is prefix-sliced to
        # tot_bits_users * NSUBC_VALID bits below, so anything trailing the
        # angles is ignored: an FCS where the adapter appends one, and the MU
        # Exclusive Beamforming Report's Delta SNR block on an MU report.
        # Trimming a fixed four bytes instead discards real feedback on the
        # adapters that append no FCS.
        if standard == "AX":
            Feedback_angles = packet_raw[(i + 62 + 2*(nc_idx + 1)):]
        if standard == "AC":
            Feedback_angles = packet_raw[(i + 58 + 2*(nc_idx + 1)):]
            
        Feedback_angles_splitted = np.array(wrap(Feedback_angles, 2))
        Feedback_angles_bin = ""

        for idx in range(0, len(Feedback_angles_splitted)):
            bin_str = str(format(hex2dec(Feedback_angles_splitted[idx]), '08b'))
            if LSB:
                bin_str = bin_str[::-1]
            Feedback_angles_bin += bin_str

        # A frame carrying fewer angle bits than its configuration calls for is
        # reported and skipped; unchecked, the short read surfaces as an empty
        # binary string inside bfi_angles() and ends the run.
        required_bits = tot_bits_users * NSUBC_VALID
        if len(Feedback_angles_bin) < required_bits:
            print(f"[!] skipping packet {p}: {pkt_config} at {pkt_bw} MHz needs {required_bits} "
                  f"angle bits, frame carries {len(Feedback_angles_bin)}", file=sys.stderr)
            continue

        Feed_back_angles_bin_chunk = np.array(wrap(Feedback_angles_bin[:required_bits], tot_bits_users))

        angle = bfi_angles(Feed_back_angles_bin_chunk, LSB, NSUBC_VALID, order_bits)

        # Reconstruct the  V-Matrix
        v_matrix = vmatrices(angle, phi_bit, psi_bit, NSUBC_VALID, Nr, Nc_users, pkt_config)
        
        # Merge the absolute timestamp with the Spatial Matrix for VSS-LMS interpolation
        buckets_v_matrices[bucket_key].append(
            (timestamp, v_matrix, rssi, stream_snr, signal_chains))
        # Merge the absolute timestamp with the raw angles for logging
        buckets_angles[bucket_key].append((timestamp, angle))

    np.save(saved_vmatrices, buckets_v_matrices)
    np.save(saved_angles, buckets_angles)
    print(f"[*] Extraction complete. Saved to {saved_vmatrices}")