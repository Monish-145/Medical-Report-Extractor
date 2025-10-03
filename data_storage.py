import os, json

# Define dedicated save folders
FINAL_DIR = "data/final_reports"
CORRECT_DIR = "data/corrections"

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(CORRECT_DIR, exist_ok=True)

def save_confirmed(filename, data):
    """
    Save parsed report JSON into data/final_reports/
    """
    base = os.path.splitext(filename)[0]
    out_path = os.path.join(FINAL_DIR, f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path

def save_correction(filename, data):
    """
    Save user-corrected JSON into data/corrections/
    """
    base = os.path.splitext(filename)[0]
    out_path = os.path.join(CORRECT_DIR, f"{base}_corrected.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path
