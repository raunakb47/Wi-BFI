import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

ANALYSIS_ROOT = "../bfi-workspace/analysis"

def load_smart_rssi(file_path):
    """Handles single-antenna 'ghost' values or true diversity."""
    raw_data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Handle '-56,-56' or '-56'
                parts = [float(x) for x in line.split(',') if x]
                raw_data.append(parts)
        
        data = np.array(raw_data)
        
        # If the Alfa reports two identical chains, just return one
        if data.ndim > 1 and data.shape[1] > 1:
            if np.allclose(data[:, 0], data[:, 1], atol=0.1):
                return data[:, 0:1] # Return as (N, 1)
        return data
    except Exception as e:
        print(f"   ! RSSI Load Error: {e}")
        return np.array([])

def generate_reports():
    # Create the analysis root if it doesn't exist
    if not os.path.exists(ANALYSIS_ROOT):
        print(f"Error: {ANALYSIS_ROOT} directory not found.")
        return

    for root, dirs, files in os.walk(ANALYSIS_ROOT):
        # We only process folders that have both pieces of the puzzle
        if "angles.npy" in files and "rssi.csv" in files:
            folder = Path(root)
            report_path = folder / "sensing_report.png"
            
            # Skip if we already did the work
            if report_path.exists(): 
                continue
            
            # Parse folder name (results_MAC_MIMO_BW)
            parts = folder.name.split('_')
            mac_label = parts[1] if len(parts) > 1 else "Unknown"
            mimo_label = parts[2] if len(parts) > 2 else ""
            bw_label = parts[3] if len(parts) > 3 else ""

            print(f"Processing: {mac_label} ({mimo_label} {bw_label})")

            try:
                # 1. Load the data
                angles = np.load(folder / "angles.npy")
                rssi_data = load_smart_rssi(folder / "rssi.csv")
                
                # 2. Align lengths (Packets might vary slightly between tools)
                min_len = min(len(angles), len(rssi_data))
                if min_len < 2:
                    print(f"   ! Skipping {mac_label}: Insufficient data points.")
                    continue
                
                # Extract the first phase angle (Phi) for the heatmap
                phi = angles[:min_len, :, 0]
                rssi_final = rssi_data[:min_len]

                # 3. Create the Plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                               gridspec_kw={'height_ratios': [3, 1]})

                # Top: BFI Phase Map (The Microscope)
                im = ax1.imshow(phi.T, aspect='auto', cmap='plasma', origin='lower')
                ax1.set_title(f"BFI Phase Map: {mac_label} | {mimo_label} {bw_label}", fontsize=14)
                ax1.set_ylabel("Subcarrier Index")
                plt.colorbar(im, ax=ax1, label="Radians")

                # Bottom: RSSI (The Binoculars)
                for c in range(rssi_final.shape[1]):
                    label = "Monitor Antenna" if rssi_final.shape[1] == 1 else f"Chain {c+1}"
                    ax2.plot(rssi_final[:, c], label=label, alpha=0.8, linewidth=1.2)
                
                ax2.set_title("Signal Strength (RSSI)")
                ax2.set_ylabel("dBm")
                ax2.set_xlabel("Packet Index")
                ax2.grid(True, alpha=0.3)
                if rssi_final.shape[1] > 1:
                    ax2.legend(loc='upper right', fontsize='x-small')

                plt.tight_layout()
                plt.savefig(report_path)
                plt.close()
                print(f"   [OK] Saved report to {folder.name}")

            except Exception as e:
                print(f"   ! Error plotting {folder.name}: {e}")

if __name__ == "__main__":
    generate_reports()
