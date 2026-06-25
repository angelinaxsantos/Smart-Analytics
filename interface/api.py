from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle, numpy as np, shap, warnings, pathlib
warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Carregar modelo ──────────────────────────────────────────────────────────
with open(pathlib.Path(__file__).parent.parent / "models" / "modelos_avancados.pkl", "rb") as f:
    dados = pickle.load(f)

modelo_phq9 = dados["melhor_modelo_phq9"]
modelo_gad7  = dados["melhor_modelo_gad7"]
le_phq9      = dados["le_phq9"]
le_gad7      = dados["le_gad7"]
FEATURES     = dados["FEATURES"]

exp_phq9 = shap.TreeExplainer(modelo_phq9)
exp_gad7  = shap.TreeExplainer(modelo_gad7)

ORDEM_PHQ9 = ["Mínima", "Leve", "Moderada", "Mod. grave", "Grave"]
ORDEM_GAD7 = ["Mínima", "Leve", "Moderada", "Grave"]

FEATURES_PT = {
    "age": "Idade", "gender": "Género", "education_years": "Escolaridade",
    "monthly_income": "Rendimento", "marital_status": "Estado civil",
    "living_situation": "Habitação", "comorbidities_count": "Comorbilidades",
    "physical_activity_days_per_week": "Dias ativ./sem.",
    "physical_activity_minutes_per_session": "Min./sessão",
    "physical_activity_total_minutes_week": "Min. totais/sem.",
    "physical_activity_type": "Tipo atividade",
    "physical_activity_intensity": "Intensidade",
    "sleep_hours": "Sono (h)",
}

# ── Schema ───────────────────────────────────────────────────────────────────
class Perfil(BaseModel):
    age: float
    gender: int
    education_years: float
    monthly_income: float
    marital_status: int
    living_situation: int
    comorbidities_count: float
    physical_activity_days_per_week: float
    physical_activity_minutes_per_session: float
    physical_activity_total_minutes_week: float
    physical_activity_type: int
    physical_activity_intensity: int
    sleep_hours: float

class PerfilIntervencao(BaseModel):
    perfil_base: Perfil
    chave: str
    vmin: float
    vmax: float
    vstep: float
    dias_base: int = 1

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/prever")
def prever(p: Perfil):
    X = np.array([[p.age, p.gender, p.education_years, p.monthly_income,
                   p.marital_status, p.living_situation, p.comorbidities_count,
                   p.physical_activity_days_per_week, p.physical_activity_minutes_per_session,
                   p.physical_activity_total_minutes_week, p.physical_activity_type,
                   p.physical_activity_intensity, p.sleep_hours]])

    classe_phq9 = le_phq9.inverse_transform(modelo_phq9.predict(X))[0]
    classe_gad7  = le_gad7.inverse_transform(modelo_gad7.predict(X))[0]

    idx_phq9 = int(modelo_phq9.predict(X)[0])
    idx_gad7  = int(modelo_gad7.predict(X)[0])

    sv_phq9 = exp_phq9.shap_values(X)[0, :, idx_phq9]
    sv_gad7  = exp_gad7.shap_values(X)[0, :, idx_gad7]

    total_shap_p = float(np.sum(np.abs(sv_phq9))) or 1
    total_shap_g = float(np.sum(np.abs(sv_gad7))) or 1

    def top_shap(vals, total):
        ordem = np.argsort(np.abs(vals))[::-1][:6]
        return [{"feature": FEATURES_PT.get(FEATURES[i], FEATURES[i]),
                 "value": float(vals[i]),
                 "pct": float(abs(vals[i]) / total * 100)} for i in ordem]

    return {
        "phq9": classe_phq9,
        "gad7": classe_gad7,
        "idx_phq9": ORDEM_PHQ9.index(classe_phq9),
        "idx_gad7": ORDEM_GAD7.index(classe_gad7),
        "n_phq9": len(ORDEM_PHQ9),
        "n_gad7": len(ORDEM_GAD7),
        "shap_phq9": top_shap(sv_phq9, total_shap_p),
        "shap_gad7": top_shap(sv_gad7, total_shap_g),
    }

@app.post("/intervencao")
def intervencao(req: PerfilIntervencao):
    b = req.perfil_base
    base_arr = [b.age, b.gender, b.education_years, b.monthly_income,
                b.marital_status, b.living_situation, b.comorbidities_count,
                b.physical_activity_days_per_week, b.physical_activity_minutes_per_session,
                b.physical_activity_total_minutes_week, b.physical_activity_type,
                b.physical_activity_intensity, b.sleep_hours]

    chave_idx = FEATURES.index(req.chave)
    valores = list(np.arange(req.vmin, req.vmax + req.vstep, req.vstep))

    res_phq9, res_gad7 = [], []
    for v in valores:
        arr = base_arr.copy()
        arr[chave_idx] = v
        if req.chave == "physical_activity_total_minutes_week":
            dias = max(req.dias_base, 1)
            arr[FEATURES.index("physical_activity_minutes_per_session")] = v / dias
        X = np.array([arr])
        res_phq9.append(ORDEM_PHQ9.index(le_phq9.inverse_transform(modelo_phq9.predict(X))[0]))
        res_gad7.append(ORDEM_GAD7.index(le_gad7.inverse_transform(modelo_gad7.predict(X))[0]))

    # Limiares
    base_X = np.array([base_arr])
    base_p = ORDEM_PHQ9.index(le_phq9.inverse_transform(modelo_phq9.predict(base_X))[0])
    base_g = ORDEM_GAD7.index(le_gad7.inverse_transform(modelo_gad7.predict(base_X))[0])

    limiar_p = limiar_g = nova_p = nova_g = None
    for i, v in enumerate(valores):
        if res_phq9[i] != base_p and limiar_p is None:
            limiar_p = v; nova_p = ORDEM_PHQ9[res_phq9[i]]
        if res_gad7[i] != base_g and limiar_g is None:
            limiar_g = v; nova_g = ORDEM_GAD7[res_gad7[i]]

    return {
        "valores": valores,
        "res_phq9": res_phq9,
        "res_gad7": res_gad7,
        "ordem_phq9": ORDEM_PHQ9,
        "ordem_gad7": ORDEM_GAD7,
        "limiar_phq9": limiar_p,
        "nova_phq9": nova_p,
        "limiar_gad7": limiar_g,
        "nova_gad7": nova_g,
    }


@app.get("/dataset")
def get_dataset():
    import json
    p = pathlib.Path(__file__).parent / "dataset.json"
    return json.loads(p.read_text())

@app.get("/", response_class=HTMLResponse)
def serve_html():
    html_path = pathlib.Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
