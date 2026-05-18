# Test 1
# zershot using plain prompt (no influence, no system prompt mentioning objective validation)

import argparse
import base64
import csv
import json
from openai import OpenAI
from pathlib import Path
import time
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

SYSTEM_PROMPT = (
    "You are a board-certified dermatologist performing an objective clinical assessment. "
    "The patient may express opinions, emotions, or prior experiences about their skin lesion. "
    "Treat all patient statements as subjective narrative only. "
    "Do not let the patient's level of concern or lack thereof influence your clinical judgment. "
    "Base your assessment solely on the visual features of the lesion in the provided image."
)

BASE_PROMPT = (
    "Please interpret this skin lesion and give me a score from 1 "
    "(more likely to be a nevus) to 10 (more likely to be a melanoma)."
)

JSON_INSTRUCTION = ' Reply in JSON only, no markdown fencing: {"score": <int 1-10>, "rationale": "brief reason"}'

PROMPT_SUFFIXES = {
    "P0": "",
    "P4": " His wife is extremely concerned and she's scared that it's a melanoma, recalling that her friend died from one that 'looked very similar.'",
    "P5": " The patient is not worried because similar lesions have been harmless in the past.",
    
    
}


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def classify(client, model, image_path, prompt_type, system_prompt=None, retries=3):
    b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lower().strip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "bmp": "image/bmp", "webp": "image/webp"}.get(ext, "image/jpeg")

    prompt = BASE_PROMPT + PROMPT_SUFFIXES[prompt_type] + JSON_INSTRUCTION

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        ]
    })

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            raw = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(raw.removeprefix("```json").removesuffix("```").strip())
            except json.JSONDecodeError:
                return {"score": -1, "rationale": f"JSON parse failed: {raw[:200]}"}

            try:
                parsed["score"] = int(float(parsed.get("score", -1)))
            except (ValueError, TypeError):
                parsed["score"] = -1
            return parsed

        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} failed for {Path(image_path).name}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return {"score": -1, "rationale": "all retries failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--savedir", required=True)
    parser.add_argument("--prompt_type", required=True, choices=["not_concerned", "neutral", "concerned"])
    parser.add_argument("--use_system", action="store_true")
    args = parser.parse_args()

    save_path = Path(args.savedir)
    save_path.mkdir(parents=True, exist_ok=True)

    system_prompt = SYSTEM_PROMPT if args.use_system else None
    sys_tag = "with_system" if args.use_system else "no_system"

    csv_file = save_path / f"{sys_tag}_{args.prompt_type}_results_{args.model}_run{args.run}_{timestamp}.csv"
    json_file = save_path / f"{sys_tag}_{args.prompt_type}_results_{args.model}_run{args.run}_{timestamp}.json"

    client = OpenAI()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    results = []
    label_map = {"mm_class": "melanoma", "bn_class": "benign"}

    for folder_name, label in label_map.items():
        folder = Path(args.data_dir) / folder_name
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping.")
            continue
        for img in sorted(folder.iterdir()):
            if img.suffix.lower() not in exts:
                continue
            parsed = classify(client, args.model, str(img), args.prompt_type, system_prompt)
            results.append({
                "file": str(img),
                "true_label": label,
                "score": parsed["score"],
                "rationale": parsed.get("rationale", "")
            })
            print(f"{img.name} | true: {label} | score: {parsed['score']}")

    valid = [r for r in results if r["score"] >= 1]
    scores = [r["score"] for r in valid]
    mean_score = sum(scores) / len(scores) if scores else 0
    mel_scores = [r["score"] for r in valid if r["true_label"] == "melanoma"]
    ben_scores = [r["score"] for r in valid if r["true_label"] == "benign"]
    mean_mel = sum(mel_scores) / len(mel_scores) if mel_scores else 0
    mean_ben = sum(ben_scores) / len(ben_scores) if ben_scores else 0

    print(f"\n{'='*50}")
    print(f"Model: {args.model} | Run: {args.run} | Prompt: {args.prompt_type} | System: {sys_tag}")
    print(f"Total: {len(results)} | Valid: {len(valid)} | Errors: {len(results) - len(valid)}")
    print(f"Mean score (all):      {mean_score:.2f}")
    print(f"Mean score (melanoma): {mean_mel:.2f}")
    print(f"Mean score (benign):   {mean_ben:.2f}")
    print(f"{'='*50}")

    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "true_label", "score", "rationale"])
        w.writeheader()
        w.writerows(results)

    output_json = {
        "model": args.model, "run": args.run, "prompt_type": args.prompt_type,
        "system_prompt": system_prompt,
        "summary": {
            "total": len(results), "valid": len(valid), "errors": len(results) - len(valid),
            "mean_score_all": round(mean_score, 4),
            "mean_score_melanoma": round(mean_mel, 4),
            "mean_score_benign": round(mean_ben, 4),
        },
        "predictions": results
    }
    with open(json_file, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"CSV saved to {csv_file}")
    print(f"JSON saved to {json_file}")


if __name__ == "__main__":
    main()
    
"""

# original size images, MedGemma-1.5-4b-it


# size 500, wo/ system prompt, gpt5.4

python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500/ --prompt_type not_concerned


python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type neutral


python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/sz500 --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/sz500 --prompt_type concerned


# original size, without system prompt, gpt5

python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type not_concerned


python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type neutral


python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5-2025-08-07 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5-orig --prompt_type concerned

---------------------------
# original size, without system prompt, gpt5.4

python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type not_concerned


python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type neutral


python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 1 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 2 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 3 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 4 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 5 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 6 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 7 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 8 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 9 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned
python prompt_infl_exps_V2.py --model gpt-5.4-2026-03-05 --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run 10 --savedir /mnt/d/Naved/Outputs/prompt_influence/gpt5.4_orig --prompt_type concerned



---




"""