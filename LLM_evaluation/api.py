"""
api.py
Servidor FastAPI que expõe endpoints para:
  - Listar modelos disponíveis
  - Prever phq9/gad7 para uma instância
  - Correr avaliação em batch sobre o dataset completo
"""

import time
import math
from pathlib import Path
from typing import Optional

import ollama
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prompt_builder import (
    load_rules,
    load_dataset,
    build_system_prompt,
    build_user_prompt,
    parse_llm_response,
    get_phq9_class,
    get_gad7_class,
)

# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────
OLLAMA_HOST  = "http://192.168.65.6:11434"
RULES_PATH   = "dataset_rules.json"
CSV_PATH     = "Dataset_Idosos.csv"

# Modelos disponíveis no servidor (deepseek-r1:70b excluído)
AVAILABLE_MODELS = [
    "deepseek-r1:32b",
    "mistral-small3.2:24b",
    "gpt-oss:20b",
    "nemotron-cascade-2:30b",
    "nemotron-3-nano:30b",
    "qwen3.6:35b",
    "qwen3.6:27b",
    "gemma4:26b",
    "granite4.1:30b",
    "gemma4:31b",
    "nemotron3:33b",
    "gemma3:4b",
    "qwen3.5:35b",
    "qwen3-vl:32b",
]

# ──────────────────────────────────────────────
# Estado global de jobs de avaliação em batch
# ──────────────────────────────────────────────
# job_id → { status, progress, total, results, metrics, errors }
jobs: dict[str, dict] = {}

# ──────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="LLM PHQ-9 / GAD-7 Predictor",
    description="Avalia LLMs locais na previsão de scores de depressão e ansiedade em idosos.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente Ollama ligado ao servidor da rede local
client = ollama.Client(host=OLLAMA_HOST)


# ──────────────────────────────────────────────
# Schemas Pydantic
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    model: str
    participant: dict  # dicionário com as colunas do CSV (sem phq9/gad7)


class BatchRequest(BaseModel):
    model: str
    max_samples: Optional[int] = None  # None = todo o dataset; int = primeiros N


class PredictResponse(BaseModel):
    model: str
    participant_id: Optional[str]
    phq9_pred: int
    phq9_class_pred: str
    gad7_pred: int
    gad7_class_pred: str
    raw_response: str


class BatchJobResponse(BaseModel):
    job_id: str
    message: str


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _load_system_prompt() -> str:
    return build_system_prompt("")


def _call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """Chama o Ollama e devolve o texto da resposta."""
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        options={"temperature": 0.0},  # determinístico
    )
    return response.message.content.strip()


def _compute_metrics(results: list[dict]) -> dict:
    """
    Calcula métricas de avaliação sobre os resultados batch.
    Trata phq9_total e gad7_total como regressão (MAE, RMSE)
    e as classes como classificação (accuracy, per-class F1).
    """
    if not results:
        return {}

    # ── Regressão ──
    phq9_errors, gad7_errors = [], []
    phq9_sq, gad7_sq = [], []

    # ── Classificação ──
    phq9_classes = set()
    gad7_classes = set()
    phq9_tp, phq9_fp, phq9_fn = {}, {}, {}
    gad7_tp, gad7_fp, gad7_fn = {}, {}, {}

    for r in results:
        true_phq9 = r["phq9_true"]
        pred_phq9 = r["phq9_pred"]
        true_gad7 = r["gad7_true"]
        pred_gad7 = r["gad7_pred"]
        true_phq9_cls = r["phq9_class_true"]
        pred_phq9_cls = r["phq9_class_pred"]
        true_gad7_cls = r["gad7_class_true"]
        pred_gad7_cls = r["gad7_class_pred"]

        phq9_errors.append(abs(true_phq9 - pred_phq9))
        gad7_errors.append(abs(true_gad7 - pred_gad7))
        phq9_sq.append((true_phq9 - pred_phq9) ** 2)
        gad7_sq.append((true_gad7 - pred_gad7) ** 2)

        for cls in [true_phq9_cls, pred_phq9_cls]:
            phq9_classes.add(cls)
            phq9_tp.setdefault(cls, 0)
            phq9_fp.setdefault(cls, 0)
            phq9_fn.setdefault(cls, 0)

        for cls in [true_gad7_cls, pred_gad7_cls]:
            gad7_classes.add(cls)
            gad7_tp.setdefault(cls, 0)
            gad7_fp.setdefault(cls, 0)
            gad7_fn.setdefault(cls, 0)

        # PHQ9 classe
        if true_phq9_cls == pred_phq9_cls:
            phq9_tp[true_phq9_cls] += 1
        else:
            phq9_fp[pred_phq9_cls] += 1
            phq9_fn[true_phq9_cls] += 1

        # GAD7 classe
        if true_gad7_cls == pred_gad7_cls:
            gad7_tp[true_gad7_cls] += 1
        else:
            gad7_fp[pred_gad7_cls] += 1
            gad7_fn[true_gad7_cls] += 1

    n = len(results)

    def per_class_f1(tp, fp, fn, classes):
        f1s = {}
        for cls in classes:
            p = tp.get(cls, 0) / max(tp.get(cls, 0) + fp.get(cls, 0), 1)
            r = tp.get(cls, 0) / max(tp.get(cls, 0) + fn.get(cls, 0), 1)
            f1s[cls] = round(2 * p * r / max(p + r, 1e-9), 4)
        return f1s

    def macro_f1(f1s):
        return round(sum(f1s.values()) / max(len(f1s), 1), 4)

    phq9_f1s = per_class_f1(phq9_tp, phq9_fp, phq9_fn, phq9_classes)
    gad7_f1s = per_class_f1(gad7_tp, gad7_fp, gad7_fn, gad7_classes)

    phq9_accuracy = sum(
        1 for r in results if r["phq9_class_true"] == r["phq9_class_pred"]
    ) / n
    gad7_accuracy = sum(
        1 for r in results if r["gad7_class_true"] == r["gad7_class_pred"]
    ) / n

    return {
        "n_evaluated": n,
        "phq9": {
            "mae":  round(sum(phq9_errors) / n, 4),
            "rmse": round(math.sqrt(sum(phq9_sq) / n), 4),
            "class_accuracy": round(phq9_accuracy, 4),
            "per_class_f1":   phq9_f1s,
            "macro_f1":       macro_f1(phq9_f1s),
        },
        "gad7": {
            "mae":  round(sum(gad7_errors) / n, 4),
            "rmse": round(math.sqrt(sum(gad7_sq) / n), 4),
            "class_accuracy": round(gad7_accuracy, 4),
            "per_class_f1":   gad7_f1s,
            "macro_f1":       macro_f1(gad7_f1s),
        },
    }


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/models", summary="Lista modelos disponíveis")
def list_models():
    return {"models": AVAILABLE_MODELS}


@app.get("/health", summary="Verifica se o servidor Ollama está acessível")
def health():
    try:
        client.list()
        return {"status": "ok", "ollama_host": OLLAMA_HOST}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama inacessível: {e}")


@app.post("/predict", response_model=PredictResponse, summary="Prevê PHQ-9 e GAD-7 para uma instância")
def predict(req: PredictRequest):
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{req.model}' não disponível.")

    system_prompt = _load_system_prompt()
    user_prompt   = build_user_prompt(req.participant)

    try:
        raw = _call_llm(req.model, system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro ao chamar Ollama: {e}")

    parsed = parse_llm_response(raw)
    if parsed is None:
        raise HTTPException(
            status_code=422,
            detail=f"Não foi possível fazer parse da resposta do LLM: {repr(raw)}",
        )

    return PredictResponse(
        model=req.model,
        participant_id=str(req.participant.get("participant_id", "")),
        raw_response=raw,
        **parsed,
    )


@app.post("/evaluate/start", response_model=BatchJobResponse, summary="Inicia avaliação batch (async)")
def start_evaluation(req: BatchRequest, background_tasks: BackgroundTasks):
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{req.model}' não disponível.")

    job_id = f"{req.model.replace(':', '_')}_{int(time.time())}"
    jobs[job_id] = {
        "status":   "running",
        "model":    req.model,
        "progress": 0,
        "total":    0,
        "results":  [],
        "metrics":  {},
        "errors":   [],
    }

    background_tasks.add_task(_run_evaluation, job_id, req.model, req.max_samples)
    return BatchJobResponse(job_id=job_id, message="Avaliação iniciada em background.")


@app.get("/evaluate/status/{job_id}", summary="Estado de um job de avaliação")
def evaluation_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    job = jobs[job_id]
    return {
        "job_id":   job_id,
        "status":   job["status"],
        "model":    job["model"],
        "progress": job["progress"],
        "total":    job["total"],
        "n_errors": len(job["errors"]),
        "metrics":  job["metrics"],  # preenchido só quando status == "done"
    }


@app.get("/evaluate/results/{job_id}", summary="Resultados detalhados de um job")
def evaluation_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return jobs[job_id]


# ──────────────────────────────────────────────
# Background task — avaliação batch
# ──────────────────────────────────────────────

def _run_evaluation(job_id: str, model: str, max_samples: Optional[int]):
    job = jobs[job_id]
    try:
        dataset = load_dataset(CSV_PATH)
        if max_samples is not None:
            dataset = dataset[:max_samples]

        job["total"] = len(dataset)
        system_prompt = _load_system_prompt()
        results = []

        for i, row in enumerate(dataset):
            try:
                user_prompt = build_user_prompt(row)
                raw = _call_llm(model, system_prompt, user_prompt)
                parsed = parse_llm_response(raw)

                if parsed is None:
                    job["errors"].append({
                        "participant_id": row.get("participant_id"),
                        "error": f"Parse falhou: {repr(raw)}",
                    })
                else:
                    true_phq9 = int(float(row["phq9_total"]))
                    true_gad7 = int(float(row["gad7_total"]))
                    results.append({
                        "participant_id":  row.get("participant_id"),
                        "phq9_true":       true_phq9,
                        "phq9_pred":       parsed["phq9_pred"],
                        "phq9_class_true": get_phq9_class(true_phq9),
                        "phq9_class_pred": parsed["phq9_class_pred"],
                        "gad7_true":       true_gad7,
                        "gad7_pred":       parsed["gad7_pred"],
                        "gad7_class_true": get_gad7_class(true_gad7),
                        "gad7_class_pred": parsed["gad7_class_pred"],
                        "raw_response":    raw,
                    })
            except Exception as e:
                job["errors"].append({
                    "participant_id": row.get("participant_id"),
                    "error": str(e),
                })

            job["progress"] = i + 1
            job["results"]  = results

        job["metrics"] = _compute_metrics(results)
        job["status"]  = "done"

    except Exception as e:
        job["status"] = "error"
        job["errors"].append({"error": str(e)})


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)