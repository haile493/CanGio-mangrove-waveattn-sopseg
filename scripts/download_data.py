import os
import sys
import gdown
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Google Drive")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "anhgoc", "cut"],
        help="Which imagery to download: 'anhgoc' (original full scenes), 'cut' (pre-cut chips), or 'all'"
    )
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent download threads")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_base = os.path.join(root_dir, "data")
    os.makedirs(target_base, exist_ok=True)
    
    url = "https://drive.google.com/drive/folders/1BkGYbF-rR25XAYGulf9pVtOATcqEwirQ?usp=sharing"
    print(f"Connecting to Google Drive and scanning folder metadata (source={args.source})...")
    items = gdown.download_folder(url=url, skip_download=True, quiet=True, use_cookies=False)
    
    tasks = []
    for item in items:
        p = item.path if hasattr(item, 'path') else str(item)
        p_lower = p.lower()
        is_anhgoc = "anhgoc" in p_lower
        is_cut = ("cut_" in p_lower or "cut\\" in p_lower or "cut/" in p_lower) and not is_anhgoc
        is_gt = "vunggaydo" in p_lower
        
        include = False
        if is_gt:
            include = True
        elif args.source == "all":
            include = True
        elif args.source == "anhgoc" and is_anhgoc:
            include = True
        elif args.source == "cut" and is_cut:
            include = True
            
        if include:
            out_path = os.path.join(target_base, p)
            file_id = item.id
            tasks.append((file_id, out_path, p))
            
    print(f"Total target files identified: {len(tasks)}")
    
    import requests
    import time

    def direct_gdrive_download(file_id: str, destination: str):
        url = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                response = session.get(url, params={"id": file_id, "confirm": value}, stream=True, timeout=60)
                break
        
        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type and response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

    def download_one(task):
        fid, out_p, rel_p = task
        min_sz = 1000 if not out_p.lower().endswith(".tif") else 1024 * 1024
        if os.path.exists(out_p) and os.path.getsize(out_p) > min_sz:
            return f"Skipped (exists): {rel_p}"
        os.makedirs(os.path.dirname(out_p), exist_ok=True)

        for attempt in range(1, 6):
            if os.path.exists(out_p) and os.path.getsize(out_p) < min_sz:
                try:
                    os.remove(out_p)
                except Exception:
                    pass
            try:
                direct_gdrive_download(fid, out_p)
                if os.path.exists(out_p) and os.path.getsize(out_p) > 0:
                    size_mb = os.path.getsize(out_p) / (1024 * 1024)
                    return f"Downloaded: {rel_p} ({size_mb:.2f} MB)"
            except Exception as e:
                if attempt == 5:
                    return f"Failed after 5 attempts ({rel_p}): {e}"
                time.sleep(2 * attempt)
        return f"Failed: {rel_p}"

    print(f"Starting parallel download with ThreadPoolExecutor (workers={args.workers})...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, t): t for t in tasks}
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            completed += 1
            print(f"[{completed}/{len(tasks)}] {res}", flush=True)

    print("Dataset download completed successfully!")

if __name__ == "__main__":
    main()
