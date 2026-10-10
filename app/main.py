"""
DeliveryRisk API
----------------
API de previsão de risco de atraso de entregas, treinada com dados reais
do dataset Olist (e-commerce brasileiro, 2016-2018).

Autor: Rafael Simões
Stack: FastAPI + scikit-learn + joblib + Docker + GCP Cloud Run
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import numpy as np
import os

from app.schema import PedidoInput, PrevisaoOutput, ESTADOS_BRASILEIROS

# ── Carregar modelo na inicialização ─────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")

try:
    payload = joblib.load(MODEL_PATH)
    MODEL = payload["model"]
    FEATURE_NAMES = payload["feature_names"]
    print(f"✓ Modelo carregado: {MODEL_PATH} | {len(FEATURE_NAMES)} features")
except Exception as e:
    raise RuntimeError(f"Falha ao carregar o modelo em '{MODEL_PATH}': {e}")


# ── Inicializar app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="DeliveryRisk API",
    description=(
        "Prevê o risco de atraso de um pedido de e-commerce **no momento da compra**, "
        "usando um modelo Random Forest treinado com 96.470 pedidos reais do Olist.\n\n"
        "**Métricas do modelo:** Recall 61% | Precisão 15.7% | Acurácia 75.2%\n\n"
        "Para operação logística, o recall é a métrica mais importante: "
        "o modelo detecta 6 de cada 10 atrasos reais antes da entrega."
    ),
    version="1.0.0",
    contact={
        "name": "Rafael Simões",
        "url": "https://github.com/rafaelsimoesdev/deliveryrisk-api",
    },
)


# ── Helpers ───────────────────────────────────────────────────────────────────
# Limiar de decisão do modelo: é nele que valem as métricas de validação.
LIMIAR_ALERTA = 0.50


def _nivel_risco(prob: float) -> str:
    """BAIXO < 0.30 <= MÉDIO < 0.50 (limiar) <= ALTO."""
    if prob < 0.30:
        return "BAIXO"
    elif prob < LIMIAR_ALERTA:
        return "MÉDIO"
    return "ALTO"


def _pedido_para_features(pedido: PedidoInput) -> pd.DataFrame:
    """Converte o payload de entrada no vetor de features esperado pelo modelo."""
    row = {
        "mes_compra": pedido.mes_compra,
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
    O modelo retorna a previsão, a probabilidade e o nível de risco.
    """
    try:
        X = _pedido_para_features(pedido)
        prob = float(MODEL.predict_proba(X)[0][1])
        risco = prob >= LIMIAR_ALERTA
        nivel = _nivel_risco(prob)

        if nivel == "ALTO":
            mensagem = (
                f"⚠️ Risco ALTO de atraso ({prob*100:.1f}%). "
                "Recomenda-se monitoramento preventivo."
            )
        elif nivel == "MÉDIO":
            mensagem = (
                f"🟡 Risco MÉDIO de atraso ({prob*100:.1f}%). "
                "Abaixo do limiar de alerta, mas vale acompanhar."
            )
        else:
            mensagem = f"✅ Risco BAIXO de atraso ({prob*100:.1f}%)."

        return PrevisaoOutput(
            risco_atraso=risco,
            probabilidade_atraso=round(prob, 4),
            nivel_risco=nivel,
            mensagem=mensagem,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")


@app.get("/model/info", tags=["Modelo"])
def model_info():
    """Informações sobre o modelo em produção."""
    importancias = dict(
        zip(FEATURE_NAMES, MODEL.feature_importances_.tolist())
    )
    top5 = dict(
        sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:5]
    )
    return {
        "tipo": type(MODEL).__name__,
        "n_estimators": MODEL.n_estimators,
        "max_depth": MODEL.max_depth,
        "n_features": len(FEATURE_NAMES),
        "metricas_validacao": {
            "acuracia": "75.2%",
            "precisao": "15.7%",
            "recall": "61.0%",
        },
        "top5_features_mais_importantes": top5,
        "dataset": "Olist Brazilian E-Commerce (2016-2018)",
        "total_pedidos_treino": 77176,
    }
