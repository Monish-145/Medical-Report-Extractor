import os, json
from glob import glob
from ocr_processor import simple_ocr
from lab_report_extractor import extract_with_text
from data_schemas import Report

EXPECTED_DIR = "data/samples"
RESULTS_DIR = "data/final_reports"

def safe_load(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def compare_dicts(expected, actual):
    results = {}
    for key, exp_val in expected.items():
        act_val = actual.get(key, "UNKNOWN")
        results[key] = (str(exp_val).lower() == str(act_val).lower())
    return results

def compare_tests(expected_tests, actual_tests):
    results = []
    total = len(expected_tests)

    for exp in expected_tests:
        found = False
        for act in actual_tests:
            if exp["name"].lower() == act["name"].lower():
                same_val = str(exp.get("value")).lower() == str(act.get("value", "")).lower()
                same_unit = str(exp.get("unit")).lower() == str(act.get("unit", "")).lower()
                results.append({
                    "name": exp["name"],
                    "value_match": same_val,
                    "unit_match": same_unit
                })
                found = True
                break
        if not found:
            results.append({
                "name": exp["name"],
                "value_match": False,
                "unit_match": False
            })

    score = sum(1 for r in results if r["value_match"] and r["unit_match"]) / max(1, total)
    return results, score

def evaluate_all():
    summary = {}
    expected_files = glob(os.path.join(EXPECTED_DIR, "*_expected.json"))

    for exp_file in expected_files:
        base = os.path.basename(exp_file).replace("_expected.json", "")
        expected = safe_load(exp_file)

        if not expected:
            summary[base] = {"status": "missing"}
            continue

        # -----------------------------------
        # Step 1: Run OCR on PNG
        # -----------------------------------
        img_file = exp_file.replace("_expected.json", ".png")
        try:
            tokens, raw_text = simple_ocr(img_file)
        except Exception as e:
            summary[base] = {"status": "ocr_failed", "error": str(e)}
            continue

        # -----------------------------------
        # Step 2: Regex-only extractor
        # -----------------------------------
        # Extract using the main extraction function
        extracted = extract_with_text(raw_text)
        regex_patient = extracted.get("patient", {})
        regex_tests = extracted.get("tests", [])

        patient_results_regex = compare_dicts(expected.get("patient", {}), regex_patient)
        test_results_regex, test_score_regex = compare_tests(expected.get("tests", []), regex_tests)
        regex_patient_score = sum(patient_results_regex.values()) / max(1, len(patient_results_regex))

        # -----------------------------------
        # Step 3: Hybrid extractor (regex + BioClinicalBERT + schemas)
        # -----------------------------------
        got = extract_with_text(raw_text)
        try:
            got = Report(**got).dict()  # normalize via schema
        except Exception:
            pass

        patient_results_hybrid = compare_dicts(expected.get("patient", {}), got.get("patient", {}))
        test_results_hybrid, test_score_hybrid = compare_tests(expected.get("tests", []), got.get("tests", []))
        hybrid_patient_score = sum(patient_results_hybrid.values()) / max(1, len(patient_results_hybrid))

        # -----------------------------------
        # Collect results
        # -----------------------------------
        summary[base] = {
            "regex": {
                "patient_accuracy": round(regex_patient_score * 100, 2),
                "test_accuracy": round(test_score_regex * 100, 2),
                "patient_results": patient_results_regex,
                "test_results": test_results_regex
            },
            "hybrid": {
                "patient_accuracy": round(hybrid_patient_score * 100, 2),
                "test_accuracy": round(test_score_hybrid * 100, 2),
                "patient_results": patient_results_hybrid,
                "test_results": test_results_hybrid
            }
        }

    return summary
