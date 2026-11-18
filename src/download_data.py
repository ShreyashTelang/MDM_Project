"""
download_data.py
================
Module to download ECG recordings from the MIT-BIH Arrhythmia Database
available on PhysioNet (https://physionet.org/content/mitdb/1.0.0/).

Supports:
- Curated subset download for fast development / academic demonstration
- Full 48-record download
- Specific individual record download
- Robust retry logic with backoff for network resilience
"""

import os
import sys
import time
import argparse
import requests
import wfdb

ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# Standard Inter-Patient Split (de Chazal et al., 2004)
DS1_TRAIN = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230'
]

DS2_TEST = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234'
]

# Curated balanced subset across DS1 (Train) and DS2 (Test)
# Designed for fast downloading while covering all 4 arrhythmia categories
CURATED_RECORDS = [
    # DS1 (Train) records: Normal, PVCs, PACs, Bundle Branch
    '101', '106', '108', '109', '118', '119', '201', '208',
    # DS2 (Test) records: Normal, PVCs, PACs, Ventricular Arrhythmias
    '100', '103', '105', '111', '200', '214'
]

BASE_URL = "https://physionet.org/files/mitdb/1.0.0/"


def download_file_direct(url, dest_path, retries=3, timeout=15):
    """Downloads a file with HTTP GET and retry mechanism."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            if r.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
                return True
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1.5 * attempt)
    return False


def download_record(rec_id, target_dir='data/raw'):
    """
    Downloads all 3 components (.hea, .dat, .atr) of an ECG record.
    Uses WFDB first, and falls back to direct PhysioNet download if needed.
    """
    os.makedirs(target_dir, exist_ok=True)
    extensions = ['.hea', '.dat', '.atr']
    
    # Check if all files exist
    all_exist = all(os.path.exists(os.path.join(target_dir, f"{rec_id}{ext}")) for ext in extensions)
    if all_exist:
        return True, "already exists"
        
    try:
        wfdb.dl_files('mitdb', target_dir, [f"{rec_id}{ext}" for ext in extensions])
        return True, "downloaded via WFDB"
    except Exception:
        # Fallback to direct HTTP download
        success_all = True
        for ext in extensions:
            dest = os.path.join(target_dir, f"{rec_id}{ext}")
            if not os.path.exists(dest):
                url = f"{BASE_URL}{rec_id}{ext}"
                ok = download_file_direct(url, dest)
                if not ok:
                    success_all = False
                    break
        if success_all:
            return True, "downloaded via Direct HTTP"
        else:
            return False, "failed to download"


def download_mitbih(target_dir='data/raw', records=None, dl_all=False):
    """Main download runner."""
    os.makedirs(target_dir, exist_ok=True)
    if dl_all:
        selected_records = ALL_RECORDS
        print(f"[*] Mode: FULL MIT-BIH DATASET ({len(selected_records)} records)")
    elif records and len(records) > 0:
        selected_records = records
        print(f"[*] Mode: SPECIFIC RECORDS ({len(selected_records)} records: {selected_records})")
    else:
        selected_records = CURATED_RECORDS
        print(f"[*] Mode: CURATED BALANCED SUBSET ({len(selected_records)} records)")
        
    print(f"[*] Target Directory: '{target_dir}'")
    print("-" * 60)
    
    results = []
    for i, rec_id in enumerate(selected_records, 1):
        print(f" [{i:02d}/{len(selected_records):02d}] Record {rec_id}...", end=' ', flush=True)
        ok, msg = download_record(rec_id, target_dir)
        if ok:
            print(f"[OK: {msg}]")
            results.append(rec_id)
        else:
            print(f"[ERROR: {msg}]")
            
    print("-" * 60)
    print(f"[+] Total Successfully Verified / Available: {len(results)} records.")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia ECG dataset from PhysioNet")
    parser.add_argument('--target_dir', type=str, default='data/raw', help="Destination directory for raw data")
    parser.add_argument('--all', action='store_true', help="Download all 48 records")
    parser.add_argument('--records', nargs='+', default=None, help="Specific record numbers (e.g. --records 100 101 200)")
    args = parser.parse_args()

    download_mitbih(target_dir=args.target_dir, records=args.records, dl_all=args.all)
