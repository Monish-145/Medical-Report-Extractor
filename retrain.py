import os, json
# Note: model_training module was removed during cleanup
# This file is kept for reference but may need updates

CORRECT_DIR = "data/corrections"
TRAIN_DIR = "data/training_data"

def corrections_to_training():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    for fname in os.listdir(CORRECT_DIR):
        if not fname.endswith("_corrected.json"):
            continue
        path = os.path.join(CORRECT_DIR, fname)
        with open(path, "r") as f:
            corrected = json.load(f)
        sample = {
            "tokens": [{"text": t, "bbox": [0,0,0,0]} for t in corrected.get("patient", {}).values()],
            "meta": {"width": 1000, "height": 1000},
            "corrected": corrected
        }
        out = os.path.join(TRAIN_DIR, fname.replace("_corrected.json", ".json"))
        with open(out, "w") as f:
            json.dump(sample, f, indent=2)
        pass

if __name__ == "__main__":
    corrections_to_training()
    # train_model() - commented out as model_training module was removed
