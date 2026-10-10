"""
DeliveryRisk API: treino, avaliação e serialização do modelo de produção.

O que este script faz:
  1. Carrega a base (base_modelo.csv), descarta o mês da compra (veja abaixo) e
     separa treino (80%) e teste (20%), com o mesmo split do Projeto 2
     (random_state=42, estratificado).
  2. Treina um Random Forest e CALIBRA as probabilidades (isotônica), para que
     "30%" signifique que cerca de 30% desses pedidos realmente atrasam.
  3. Define os limiares de risco usando só dados de treino (predições
     out-of-fold), sem olhar o teste.
  4. Avalia no teste contra baselines (ROC-AUC, PR-AUC, lift, calibração).
  5. Salva o modelo em model/model.pkl e os gráficos em docs/.

Por que o mês da compra não entra: no Projeto 2 ele aparecia como a feature mais
importante, mas a validação temporal (train/validacao_temporal.py) mostrou que
ele não ajuda a prever períodos futuros. O modelo só decorava os meses de pico
de 2017-2018. Sem ele, o resultado no futuro é igual ou melhor.

Como rodar (da raiz do projeto, com base_modelo.csv na raiz):
    pip install -r requirements.txt -r train/requirements-train.txt
    python train/treinar_modelo.py

Use as MESMAS versões do requirements.txt: o model.pkl só é garantido nelas.
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RECALL_ALVO = 0.60       # ponto de operação: detectar ~6 de cada 10 atrasos
FEATURES_EXCLUIDAS = ["mes_compra"]  # não generaliza no tempo (validacao_temporal.py)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(RAIZ, "base_modelo.csv")
OUTPUT_PATH = os.path.join(RAIZ, "model", "model.pkl")
DOCS = os.path.join(RAIZ, "docs")


def novo_rf():
    # Mesmos hiperparâmetros do Projeto 2
    return RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )


def novo_calibrado(cv):
    return CalibratedClassifierCV(novo_rf(), method="isotonic", cv=cv)


def metricas(y, proba, limiar):
    pred = (proba >= limiar).astype(int)
    return {
        "acuracia": round(float(accuracy_score(y, pred)), 4),
        "precisao": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "alertas_pct": round(float(pred.mean()), 4),
    }


# ── 1. Dados ─────────────────────────────────────────────────────────────────
base = pd.read_csv(CSV_PATH)
X = base.drop(columns=["atrasou"] + FEATURES_EXCLUIDAS)
y = base["atrasou"]
FEATURE_NAMES = list(X.columns)
TAXA_BASE = float(y.mean())
LIMIAR_ALTO = round(2 * TAXA_BASE, 3)  # risco ALTO = pelo menos 2x a taxa média de atraso

print(f"Base: {len(base):,} pedidos | {len(FEATURE_NAMES)} features | atraso {TAXA_BASE:.1%}")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ── 2. Limiares a partir do treino (predições out-of-fold) ───────────────────
print("\nDefinindo limiares com validação cruzada no treino...")
oof = cross_val_predict(
    novo_calibrado(cv=3), X_treino, y_treino,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    method="predict_proba",
)[:, 1]
prec_c, rec_c, lim_c = precision_recall_curve(y_treino, oof)
candidatos = np.where(rec_c[:-1] >= RECALL_ALVO)[0]
LIMIAR_ALERTA = float(lim_c[candidatos[-1]])
print(f"Limiar de alerta (recall >= {RECALL_ALVO:.0%} no treino): {LIMIAR_ALERTA:.3f}")

# ── 3. Modelo final (Random Forest calibrado) ────────────────────────────────
print("\nTreinando o modelo final...")
modelo = novo_calibrado(cv=5).fit(X_treino, y_treino)
proba = modelo.predict_proba(X_teste)[:, 1]

# ── 4. Avaliação no teste, contra baselines ──────────────────────────────────
baseline = DummyClassifier(strategy="prior").fit(X_treino, y_treino)
p_base = baseline.predict_proba(X_teste)[:, 1]

logistica = make_pipeline(
    StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
).fit(X_treino, y_treino)
p_log = logistica.predict_proba(X_teste)[:, 1]

rf_puro = novo_rf().fit(X_treino, y_treino)
p_rf = rf_puro.predict_proba(X_teste)[:, 1]

resultados = {
    "nunca_atrasa": metricas(y_teste, p_base, 0.5),
    "regressao_logistica": metricas(y_teste, p_log, 0.5),
    "random_forest_sem_calibrar": metricas(y_teste, p_rf, 0.5),
    "modelo_final": metricas(y_teste, proba, LIMIAR_ALERTA),
}

print(f"\n{'Modelo':<28}{'Acur.':>7}{'Prec.':>7}{'Recall':>8}{'ROC':>7}{'PR-AUC':>8}")
for nome, m in resultados.items():
    print(f"{nome:<28}{m['acuracia']:>7.3f}{m['precisao']:>7.3f}{m['recall']:>8.3f}"
          f"{m['roc_auc']:>7.3f}{m['pr_auc']:>8.3f}")

brier = brier_score_loss(y_teste, proba)
brier_ref = brier_score_loss(y_teste, np.full(len(y_teste), y_treino.mean()))
print(f"\nBrier (calibração): modelo {brier:.4f} | taxa média constante {brier_ref:.4f}")

# Lift: se a operação olhar só os k% pedidos mais arriscados
ordem = np.argsort(-proba)
y_ord = y_teste.values[ordem]
lift = {}
print("\nSe a operação olhar só os pedidos mais arriscados:")
for k in (0.05, 0.10, 0.20, 0.30):
    n = int(len(y_ord) * k)
    cap = float(y_ord[:n].sum() / y_ord.sum())
    lift[f"top_{int(k * 100)}pct"] = {
        "atrasos_capturados": round(cap, 4),
        "precisao": round(float(y_ord[:n].mean()), 4),
    }
    print(f"  top {int(k * 100):>2}% -> captura {cap:.0%} dos atrasos")

# Faixas de risco: taxa real de atraso em cada uma
nivel = np.where(proba >= LIMIAR_ALTO, "ALTO",
                 np.where(proba >= LIMIAR_ALERTA, "MÉDIO", "BAIXO"))
faixas = {}
print("\nFaixas de risco no teste:")
for n_ in ("BAIXO", "MÉDIO", "ALTO"):
    mask = nivel == n_
    faixas[n_] = {
        "pedidos_pct": round(float(mask.mean()), 4),
        "taxa_real_atraso": round(float(y_teste.values[mask].mean()), 4),
    }
    print(f"  {n_:<6} {mask.mean():>6.1%} dos pedidos | atraso real {y_teste.values[mask].mean():.1%}")

# ── 5. Gráficos ──────────────────────────────────────────────────────────────
os.makedirs(DOCS, exist_ok=True)
SUPERFICIE, TEXTO, TEXTO_2 = "#fcfcfb", "#0b0b0b", "#52514e"
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#9a9992"


def estilo(ax):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(CINZA)
    ax.tick_params(colors=TEXTO_2)
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


# 5a. Curva precisão x recall
pr, rc, _ = precision_recall_curve(y_teste, proba)
pr_l, rc_l, _ = precision_recall_curve(y_teste, p_log)
m = resultados["modelo_final"]

fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SUPERFICIE)
estilo(ax)
ax.plot(rc, pr, color=AZUL, linewidth=2, label=f"Random Forest calibrado (PR-AUC {m['pr_auc']:.2f})")
ax.plot(rc_l, pr_l, color=CINZA, linewidth=2, linestyle="--",
        label=f"Regressão logística (PR-AUC {resultados['regressao_logistica']['pr_auc']:.2f})")
ax.axhline(TAXA_BASE, color=TEXTO_2, linewidth=1.2, linestyle=":",
           label=f"Modelo ingênuo (PR-AUC {TAXA_BASE:.2f})")
ax.scatter([m["recall"]], [m["precisao"]], s=70, color=LARANJA, zorder=5,
           edgecolor=SUPERFICIE, linewidth=2,
           label=f"Ponto de operação (recall {m['recall']:.0%}, precisão {m['precisao']:.0%})")
ax.set_xlabel("Recall (atrasos detectados)", color=TEXTO)
ax.set_ylabel("Precisão (alertas corretos)", color=TEXTO)
ax.set_title("Troca entre precisão e recall em cada limiar", color=TEXTO, loc="left", fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.7)
ax.legend(frameon=False, labelcolor=TEXTO, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "curva_precisao_recall.png"), dpi=150, facecolor=SUPERFICIE)
plt.close(fig)

# 5b. Calibração: probabilidade prevista x taxa real (faixas com o mesmo nº de pedidos)
def curva_calibracao(p, n_faixas=8):
    cortes_q = np.quantile(p, np.linspace(0, 1, n_faixas + 1)[1:-1])
    faixa = np.digitize(p, np.unique(cortes_q))
    prev_, real_ = [], []
    for i in np.unique(faixa):
        mask = faixa == i
        prev_.append(p[mask].mean())
        real_.append(y_teste.values[mask].mean())
    return prev_, real_


prev, real = curva_calibracao(proba)
prev_rf, real_rf = curva_calibracao(p_rf)

fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=SUPERFICIE)
estilo(ax)
ax.plot([0, 0.7], [0, 0.7], color=TEXTO_2, linewidth=1.2, linestyle=":", label="Calibração perfeita")
ax.plot(prev_rf, real_rf, color=CINZA, linewidth=2, marker="o", markersize=7,
        markeredgecolor=SUPERFICIE, markeredgewidth=2, label="Random Forest sem calibrar")
ax.plot(prev, real, color=AZUL, linewidth=2, marker="o", markersize=7,
        markeredgecolor=SUPERFICIE, markeredgewidth=2, label="Random Forest calibrado (modelo final)")
ax.set_xlabel("Probabilidade prevista pelo modelo", color=TEXTO)
ax.set_ylabel("Taxa real de atraso nesses pedidos", color=TEXTO)
ax.set_title("A probabilidade devolvida pela API é confiável?", color=TEXTO, loc="left", fontsize=12)
ax.set_xlim(0, 0.7)
ax.set_ylim(0, 0.25)
ax.legend(frameon=False, labelcolor=TEXTO, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "calibracao.png"), dpi=150, facecolor=SUPERFICIE)
plt.close(fig)

# ── 6. Importância das features (média dos modelos da calibração) ────────────
importancias = np.mean(
    [c.estimator.feature_importances_ for c in modelo.calibrated_classifiers_], axis=0
)

# ── 7. Salvar ────────────────────────────────────────────────────────────────
payload = {
    "model": modelo,
    "feature_names": FEATURE_NAMES,
    "importancias": dict(zip(FEATURE_NAMES, importancias.tolist())),
    "limiar_alerta": LIMIAR_ALERTA,
    "limiar_alto": LIMIAR_ALTO,
    "taxa_base": TAXA_BASE,
    "n_treino": int(len(X_treino)),
    "metricas_teste": resultados["modelo_final"],
    "sklearn_version": sklearn.__version__,
}
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
joblib.dump(payload, OUTPUT_PATH, compress=3)

with open(os.path.join(DOCS, "metricas.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "resultados": resultados, "lift": lift, "faixas": faixas,
            "limiar_alerta": round(LIMIAR_ALERTA, 4), "limiar_alto": LIMIAR_ALTO,
            "taxa_base": round(TAXA_BASE, 4), "brier": round(float(brier), 4),
            "brier_referencia": round(float(brier_ref), 4),
            "sklearn_version": sklearn.__version__,
        },
        f, ensure_ascii=False, indent=2,
    )

carregado = joblib.load(OUTPUT_PATH)
print(f"\nModelo salvo em {OUTPUT_PATH} (scikit-learn {carregado['sklearn_version']})")
print(f"Teste de carga OK: {len(carregado['feature_names'])} features")
