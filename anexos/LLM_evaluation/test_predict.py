"""
test_predict.py
Avalia um LLM no conjunto de teste exacto exportado dos notebooks (test_indices.json),
com paralelismo para acelerar.

Uso:
    python test_predict.py --model gpt-oss:20b
    python test_predict.py --model gpt-oss:20b --n 50   # amostra rápida
    python test_predict.py --model gpt-oss:20b --workers 5
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
import pandas as pd

from prompt_builder import (
    build_system_prompt,
    build_user_prompt,
    parse_llm_response,
    get_phq9_class,
    get_gad7_class,
)

OLLAMA_HOST      = "http://192.168.65.6:11434"
CSV_PATH         = "Dataset_Idosos.csv"
TEST_INDICES_PATH = "test_indices.json"


# ── Carregar índices exactos dos notebooks ────────────────────────────────────

def get_test_indices() -> tuple[list[int], list[int]]:
    with open(TEST_INDICES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["phq9"], data["gad7"]


# ── Inferência ────────────────────────────────────────────────────────────────

def predict_one(args):
    pos, row, model, system_prompt = args
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": build_user_prompt(row)},
            ],
            options={"temperature": 0.0},
        )
        raw    = response.message.content.strip()
        parsed = parse_llm_response(raw)
        if parsed is None:
            return pos, None, f"Parse falhou: {repr(raw[:80])}"
        return pos, parsed, None
    except Exception as e:
        return pos, None, str(e)


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], target: str) -> dict:
    n = len(results)
    if n == 0:
        return {}

    true_key       = f"{target}_true"
    pred_key       = f"{target}_pred"
    cls_true_key   = f"{target}_class_true"
    cls_pred_key   = f"{target}_class_pred"

    errors = [abs(r[true_key] - r[pred_key]) for r in results]
    sq     = [(r[true_key] - r[pred_key]) ** 2 for r in results]
    acc    = sum(1 for r in results if r[cls_true_key] == r[cls_pred_key]) / n

    classes = set(r[cls_true_key] for r in results) | set(r[cls_pred_key] for r in results)
    f1s = []
    for cls in classes:
        tp = sum(1 for r in results if r[cls_true_key] == cls and r[cls_pred_key] == cls)
        fp = sum(1 for r in results if r[cls_true_key] != cls and r[cls_pred_key] == cls)
        fn = sum(1 for r in results if r[cls_true_key] == cls and r[cls_pred_key] != cls)
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1s.append(2 * p * r / max(p + r, 1e-9))

    return {
        "n":        n,
        "mae":      round(sum(errors) / n, 4),
        "rmse":     round(math.sqrt(sum(sq) / n), 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(sum(f1s) / max(len(f1s), 1), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="gpt-oss:20b")
    parser.add_argument("--n",       type=int, default=None,
                        help="Nº de instâncias por target para teste rápido (omitir = todos)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Pedidos paralelos ao Ollama (default: 3)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Modelo  : {args.model}")
    print(f"  Workers : {args.workers}")
    print(f"  Ollama  : {OLLAMA_HOST}")
    print(f"{'='*60}\n")

    df = pd.read_csv(CSV_PATH, sep=";")
    idx_phq9, idx_gad7 = get_test_indices()

    if args.n is not None:
        idx_phq9 = idx_phq9[:args.n]
        idx_gad7 = idx_gad7[:args.n]

    # União de índices únicos — cada instância é chamada só uma vez
    all_idx = sorted(set(idx_phq9) | set(idx_gad7))
    print(f"Conjunto de teste: PHQ9={len(idx_phq9)} | GAD7={len(idx_gad7)} | "
          f"Instâncias únicas={len(all_idx)}\n")

    system_prompt = build_system_prompt("")
    tasks = [
        (pos, df.loc[idx].to_dict(), args.model, system_prompt)
        for pos, idx in enumerate(all_idx)
    ]

    cache:      dict[int, dict] = {}  # pos → resultado
    all_errors: list[dict]      = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(predict_one, t): (t[0], all_idx[t[0]]) for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            pos, df_idx = futures[future]
            _, parsed, error = future.result()
            row = df.loc[df_idx]
            pid = row.get("participant_id", str(df_idx))

            elapsed   = time.time() - start
            avg_time  = elapsed / completed
            remaining = avg_time * (len(tasks) - completed)
            eta       = f"{int(remaining//60)}m{int(remaining%60)}s"

            if error:
                print(f"[{completed}/{len(tasks)}] ID {pid} ERRO → {error[:60]}")
                all_errors.append({"participant_id": str(pid), "error": error})
            else:
                true_phq9 = int(float(row["phq9_total"]))
                true_gad7 = int(float(row["gad7_total"]))
                phq9_ok   = "✓" if parsed["phq9_pred"] == true_phq9 else "✗"
                gad7_ok   = "✓" if parsed["gad7_pred"] == true_gad7 else "✗"
                print(
                    f"[{completed}/{len(tasks)}] ETA {eta} | "
                    f"PHQ9 real={true_phq9} pred={parsed['phq9_pred']} {phq9_ok} | "
                    f"GAD7 real={true_gad7} pred={parsed['gad7_pred']} {gad7_ok}"
                )
                cache[pos] = {
                    "participant_id":  str(pid),
                    "phq9_true":       true_phq9,
                    "phq9_pred":       parsed["phq9_pred"],
                    "phq9_class_true": get_phq9_class(true_phq9),
                    "phq9_class_pred": parsed["phq9_class_pred"],
                    "gad7_true":       true_gad7,
                    "gad7_pred":       parsed["gad7_pred"],
                    "gad7_class_true": get_gad7_class(true_gad7),
                    "gad7_class_pred": parsed["gad7_class_pred"],
                }

    # Mapear posição → df_idx para separar os resultados correctamente
    pos_to_dfidx = {pos: all_idx[pos] for pos in range(len(all_idx))}
    dfidx_to_pos = {v: k for k, v in pos_to_dfidx.items()}

    results_phq9 = [cache[dfidx_to_pos[i]] for i in idx_phq9 if dfidx_to_pos.get(i) in cache]
    results_gad7 = [cache[dfidx_to_pos[i]] for i in idx_gad7 if dfidx_to_pos.get(i) in cache]

    m_phq9 = compute_metrics(results_phq9, "phq9")
    m_gad7 = compute_metrics(results_gad7, "gad7")
    total_time = time.time() - start

    print(f"\n{'='*60}")
    print(f"  RESULTADOS — {args.model}")
    print(f"  Tempo total: {total_time/60:.1f} min | Erros: {len(all_errors)}")
    print(f"{'='*60}")
    if m_phq9:
        print(f"\n  PHQ-9 (n={m_phq9['n']})")
        print(f"    MAE      : {m_phq9['mae']}")
        print(f"    RMSE     : {m_phq9['rmse']}")
        print(f"    Accuracy : {m_phq9['accuracy']*100:.1f}%")
        print(f"    Macro F1 : {m_phq9['macro_f1']}")
    if m_gad7:
        print(f"\n  GAD-7 (n={m_gad7['n']})")
        print(f"    MAE      : {m_gad7['mae']}")
        print(f"    RMSE     : {m_gad7['rmse']}")
        print(f"    Accuracy : {m_gad7['accuracy']*100:.1f}%")
        print(f"    Macro F1 : {m_gad7['macro_f1']}")
    print(f"\n{'='*60}\n")

    out_file = f"results_{args.model.replace(':', '_').replace('.', '_')}_{len(cache)}inst.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model":        args.model,
            "metrics_phq9": m_phq9,
            "metrics_gad7": m_gad7,
            "results_phq9": results_phq9,
            "results_gad7": results_gad7,
            "errors":       all_errors,
        }, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados em: {out_file}\n")


if __name__ == "__main__":
    main()