import hashlib
from pathlib import Path
import urllib.request
import torch
import platformdirs

EXPECTED_HASH = "9681ec0bab91653c04d75934118d1289820755bb3909c9ef086b900b34e62a8a"

def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash

def get_and_load_pth(url: str, app_name: str = "SWLoss") -> dict:
    try:
        data_dir = Path(platformdirs.user_data_dir(appname=app_name))
    except Exception:
        # Fallback to a hidden folder in the user's home directory if platformdirs fails
        data_dir = Path.home() / f".{app_name}"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from the URL
    filename = url.split("/")[-1]
    if not filename.endswith(".pth"):
        filename += ".pth"
        
    file_path = data_dir / filename

    # 2. Check if the file already exists and is valid
    if file_path.exists():
        print(f"Found local file at: {file_path}")
        print("Verifying integrity...")
        if verify_sha256(file_path, EXPECTED_HASH):
            print("Hash verified successfully. Loading model...")
            return torch.load(file_path, weights_only=True)
        else:
            print("Local file is corrupted or outdated (hash mismatch). Redownloading...")
            file_path.unlink() # Delete the bad file

    # 3. Download the file if it doesn't exist or failed verification
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, file_path)
    except Exception as e:
        if file_path.exists():
            file_path.unlink() # Clean up partial downloads on failure
        raise RuntimeError(f"Download failed: {e}")

    # 4. Verify the newly downloaded file's hash
    print("Verifying downloaded file integrity...")
    if not verify_sha256(file_path, EXPECTED_HASH):
        file_path.unlink() # Don't keep corrupted files
        raise ValueError("Downloaded file SHA-256 hash does not match the expected hash! File deleted.")

    print("Download complete and hash verified. Loading model...")
    return torch.load(file_path, weights_only=True)

# Example Usage:
# url = "https://example.com/weights/model_weights.pth"
# model_state_dict = get_and_load_pth(url)