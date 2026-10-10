"""
DeliveryRisk API
----------------
API de previsão de risco de atraso de entregas, treinada com dados reais
do dataset Olist (e-commerce brasileiro, 2016-2018).

Autor: Rafael Simões
Stack: FastAPI + scikit-learn + joblib + Docker + GCP Cloud Run
"""

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import sklearn
import os

from app.schema import PedidoInput, PrevisaoOutput, ESTADOS_BRASILEIROS

# ── Carregar modelo na inicialização ─────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")

try:
    payload = joblib.load(MODEL_PATH)
    MODEL = payload["model"]
    FEATURE_NAMES = payload["feature_names"]
    LIMIAR_ALERTA = payload["limiar_alerta"]  # a partir daqui: risco MÉDIO (alerta)
    LIMIAR_ALTO = payload["limiar_alto"]      # a partir daqui: risco ALTO
    TAXA_BASE = payload["taxa_base"]          # taxa média de atraso na base
    METRICAS = payload["metricas_teste"]
    print(f"✓ Modelo carregado: {MODEL_PATH} | {len(FEATURE_NAMES)} features")
except Exception as e:
    raise RuntimeError(f"Falha ao carregar o modelo em '{MODEL_PATH}': {e}")

if payload.get("sklearn_version") != sklearn.__version__:
    print(
        f"⚠️ Modelo treinado com scikit-learn {payload.get('sklearn_version')}, "
        f"mas a API roda com {sklearn.__version__}. Retreine com o requirements.txt."
    )


# ── Inicializar app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="DeliveryRisk API",
    description=(
        "Prevê o risco de atraso de um pedido de e-commerce **no momento da compra**, "
        "usando um Random Forest calibrado, treinado com 96.470 pedidos reais da Olist "
        "(2016-2018).\n\n"
        f"**Validação (20% dos pedidos, separados antes do treino):** "
        f"recall {METRICAS['recall']:.0%} | precisão {METRICAS['precisao']:.0%} | "
        f"ROC-AUC {METRICAS['roc_auc']:.2f} | PR-AUC {METRICAS['pr_auc']:.2f} "
        f"(um modelo ingênuo tem PR-AUC {TAXA_BASE:.2f}).\n\n"
        "A probabilidade devolvida é calibrada: 20% significa que cerca de 20% dos "
        "pedidos com esse perfil atrasaram. A taxa média de atraso é "
        f"{TAXA_BASE:.1%}. O mês da compra não é usado: a validação temporal mostrou "
        "que ele não generaliza. Projeto de portfólio, treinado com dados históricos."
    ),
    version="1.1.0",
    contact={
        "name": "Rafael Simões",
        "url": "https://github.com/rafaelsimoesdev/deliveryrisk-api",
    },
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _nivel_risco(prob: float) -> str:
    """BAIXO abaixo do limiar de alerta; ALTO a partir do limiar alto."""
    if prob >= LIMIAR_ALTO:
        return "ALTO"
    if prob >= LIMIAR_ALERTA:
        return "MÉDIO"
    return "BAIXO"


def _mensagem(prob: float, nivel: str) -> str:
    pct = f"{prob * 100:.1f}%".replace(".", ",")
    media = f"{TAXA_BASE * 100:.1f}%".replace(".", ",")
    if nivel == "ALTO":
        vezes = f"{prob / TAXA_BASE:.1f}".replace(".", ",")
        return (
            f"🚨 Risco ALTO de atraso: {pct} de probabilidade, cerca de {vezes}x a "
            f"média ({media}). Recomenda-se ação preventiva neste pedido."
        )
    if nivel == "MÉDIO":
        return (
            f"⚠️ Risco MÉDIO de atraso: {pct} de probabilidade (média geral: {media}). "
            "Vale acompanhar este pedido."
        )
    return (
        f"✅ Risco BAIXO de atraso: {pct} de probabilidade (média geral: {media}). "
        "Nenhum alerta para este pedido."
    )


def _pedido_para_features(pedido: PedidoInput) -> pd.DataFrame:
    """Converte o payload de entrada no vetor de features esperado pelo modelo."""
    row = {
        "dia_semana_compra": pedido.dia_semana_compra,
        "prazo_prometido_dias": pedido.prazo_prometido_dias,
        "qtd_itens": pedido.qtd_itens,
        "valor_produtos": pedido.valor_produtos,
        "valor_frete": pedido.valor_frete,
        "peso_total_g": pedido.peso_total_g,
        "volume_total_cm3": pedido.volume_total_cm3,
    }
    # One-hot encoding dos estados (27 colunas, todas 0 exceto a do cliente)
    for estado in ESTADOS_BRASILEIROS:
        row[f"estado_{estado}"] = 1 if pedido.estado_cliente == estado else 0

    df = pd.DataFrame([row])
    # Garante a ordem exata das features do modelo
    return df[FEATURE_NAMES]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    """Verifica se a API está no ar."""
    return {
        "status": "ok",
        "api": "DeliveryRisk API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Status"])
def health():
    """Health check para o Cloud Run."""
    return {"status": "healthy", "modelo_carregado": True}


@app.post("/predict", response_model=PrevisaoOutput, tags=["Previsão"])
def predict(pedido: PedidoInput):
    """
    Retorna o risco de atraso de um pedido.

    Envie os dados do pedido disponíveis no momento da compra.
    O modelo retorna a probabilidade calibrada de atraso, o nível de risco
    (BAIXO, MÉDIO ou ALTO) e se há alerta (`risco_atraso`: nível MÉDIO ou ALTO).
    """
    try:
        X = _pedido_para_features(pedido)
        prob = float(MODEL.predict_proba(X)[0][1])
        nivel = _nivel_risco(prob)
        risco = nivel != "BAIXO"  # alerta = nível MÉDIO ou ALTO

        return PrevisaoOutput(
            risco_atraso=risco,
            probabilidade_atraso=round(prob, 4),
            nivel_risco=nivel,
            mensagem=_mensagem(prob, nivel),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")


@app.get("/model/info", tags=["Modelo"])
def model_info():
    """Informações sobre o modelo em produção."""
    importancias = payload["importancias"]
    top5 = dict(
        sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:5]
    )
    return {
        "tipo": "RandomForestClassifier calibrado (isotônico)",
        "n_estimators": 100,
        "max_depth": 8,
        "n_features": len(FEATURE_NAMES),
        "limiares": {
            "alerta_medio": round(LIMIAR_ALERTA, 4),
            "alto": LIMIAR_ALTO,
            "taxa_media_atraso": round(TAXA_BASE, 4),
        },
        "metricas_validacao": METRICAS,
        "top5_features_mais_importantes": top5,
        "dataset": "Olist Brazilian E-Commerce (2016-2018)",
        "total_pedidos_treino": payload["n_treino"],
        "scikit_learn": payload["sklearn_version"],
    }
