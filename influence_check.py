import json
import glob
import re
import os
import openpyxl

targets = [
    "ISIC_8073555.jpg", "ISIC_5969724.jpg", "ISIC_3704157.jpg", "ISIC_9335652.jpg", "ISIC_0021988.jpg"
]

rows = []
base = "/mnt/d/Naved/Outputs/prompt_influence/gpt5-2lvl-orig"

for path in glob.glob(f"{base}/no_system_*_results_gpt-5-2025-08-07_run*_2026-04-06.json"):
    fname = os.path.basename(path)
    m = re.match(r"(.+?)_system_(.+?)_results_gpt-5-2025-08-07_run(\d+)_2026-04-06\.json", fname)
    if not m:
        continue
    system, prompt_type, run = m.group(1), m.group(2), m.group(3)
    if prompt_type not in ("B1", "M1"):
        continue
    with open(path) as f:
        data = json.load(f)
    for entry in data["predictions"]:
        if any(t in entry["file"] for t in targets):
            rows.append([system, prompt_type, run, entry["file"], entry["true_label"], entry["score"], entry["rationale"]])

rows.sort(key=lambda x: (x[1], int(x[2])))

wb = openpyxl.Workbook()
ws = wb.active
ws.append(["system", "prompt_type", "run", "file", "true_label", "score", "rationale"])
for row in rows:
    ws.append(row)

wb.save(f"{base}/gpt5-2lvl-orig_no_system_gpt-5-2025-08-07_extracted_results_2026-04-06.xlsx")

print(f"Extracted {len(rows)} rows from {len(targets)} target file(s)")