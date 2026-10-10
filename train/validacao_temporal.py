"""
DeliveryRisk API: validação temporal ("treinar no passado, testar no futuro").

Por que existe: a base_modelo.csv só guarda o MÊS da compra, sem o ano. Numa
divisão aleatória, o modelo vê pedidos de todos os períodos no treino e pode
estar apenas "decorando o calendário" de 2017-2018 (por exemplo, os meses em
que houve picos de atraso). Aqui os pedidos são ordenados pela data real de
compra: treina nos primeiros 80% e testa nos últimos 20%, como seria em produção.

Entradas (coloque na raiz do projeto, ou passe os caminhos como argumentos):
  - base_modelo.csv              (gerada na etapa 2 do Projeto 2)
  - olist_orders_dataset.csv     (arquivo original da Olist, do Kaggle)

Uso:
    python train/validacao_temporal.py [base_modelo.csv] [olist_orders_dataset.csv]

A base_modelo.csv foi gerada filtrando os pedidos entregues, na mesma ordem do
arquivo de pedidos. Por isso as linhas são alinhadas e o script confere o
alinhamento (atraso, mês, dia da semana e prazo) antes de qualquer análise.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEED = 42
RECALL_ALVO = 0.60
FRACAO_TREINO = 0.80

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "base_modelo.csv")
PEDIDOS_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RAIZ, "olist_orders_dataset.csv")
DOCS = os.path.join(RAIZ, "docs")


def novo_calibrado(cv):
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    return CalibratedClassifierCV(rf, method="isotonic", cv=cv)


# ── 1. Recuperar a data de cada linha da base ────────────────────────────────
base = pd.read_csv(BASE_PATH)
pedidos = pd.read_csv(PEDIDOS_PATH)
for col in ("order_purchase_timestamp", "order_delivered_customer_date",
            "order_estimated_delivery_date"):
    pedidos[col] = pd.to_datetime(pedidos[col])

# Mesmo filtro da etapa 2 do Projeto 2
entregues = pedidos[
    (pedidos["order_status"] == "delivered")
    & pedidos["order_delivered_customer_date"].notna()
    & pedidos["order_estimated_delivery_date"].notna()
].reset_index(drop=True)

assert len(entregues) == len(base), (
    f"A base tem {len(base)} linhas, mas o filtro de pedidos entregues deu "
    f"{len(entregues)}. Use o olist_orders_dataset.csv original e completo."
)

compra = entregues["order_purchase_timestamp"]
prometido = entregues["order_estimated_delivery_date"].dt.normalize()
confere = (
    ((entregues["order_delivered_customer_date"].dt.normalize() > prometido).astype(int)
     == base["atrasou"]).all()
    and (compra.dt.month == base["mes_compra"]).all()
    and (compra.dt.dayofweek == base["dia_semana_compra"]).all()
    and ((prometido - compra.dt.normalize()).dt.days == base["prazo_prometido_dias"]).all()
)
assert confere, "As linhas da base não alinham com os pedidos. Abortando."
print(f"Alinhamento conferido: {len(base):,} pedidos com data de compra.")

base = base.assign(data_compra=compra.values).sort_values("data_compra", kind="stable")
base = base.reset_index(drop=True)

# ── 2. O que a linha do tempo mostra ─────────────────────────────────────────
base["ano_mes"] = base["data_compra"].dt.to_period("M").astype(str)
mensal = base.groupby("ano_mes")["atrasou"].agg(["size", "mean"])
mensal = mensal[mensal["size"] >= 100]
print("\nTaxa de atraso por mês (meses com 100+ pedidos):")
for mes, linha in mensal.iterrows():
    print(f"  {mes}: {linha['mean']:>6.1%}  ({int(linha['size']):,} pedidos)")

# ── 3. Divisão temporal ──────────────────────────────────────────────────────
corte = int(len(base) * FRACAO_TREINO)
treino, teste = base.iloc[:corte], base.iloc[corte:]
print(f"\nTreino: {treino['data_compra'].min():%Y-%m-%d} a {treino['data_compra'].max():%Y-%m-%d}"
      f" | {len(treino):,} pedidos | atraso {treino['atrasou'].mean():.1%}")
print(f"Teste:  {teste['data_compra'].min():%Y-%m-%d} a {teste['data_compra'].max():%Y-%m-%d}"
      f" | {len(teste):,} pedidos | atraso {teste['atrasou'].mean():.1%}")

colunas_aux = ["data_compra", "ano_mes", "atrasou"]
resultados = {}


def avaliar(nome, features):
    X_tr, y_tr = treino[features], treino["atrasou"]
    X_te, y_te = teste[features], teste["atrasou"]

    # Limiar escolhido só com o passado (recall alvo nas predições out-of-fold)
    oof = cross_val_predict(
        novo_calibrado(cv=3), X_tr, y_tr,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba",
    )[:, 1]
    _, rec, lim = precision_recall_curve(y_tr, oof)
    limiar = float(lim[np.where(rec[:-1] >= RECALL_ALVO)[0][-1]])

    modelo = novo_calibrado(cv=5).fit(X_tr, y_tr)
    p = modelo.predict_proba(X_te)[:, 1]
    pred = (p >= limiar).astype(int)
    resultados[nome] = {
        "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
        "pr_auc": round(float(average_precision_score(y_te, p)), 4),
        "limiar": round(limiar, 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "precisao": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "alertas_pct": round(float(pred.mean()), 4),
        "prob_media_prevista": round(float(p.mean()), 4),
        "taxa_real_teste": round(float(y_te.mean()), 4),
        "taxa_real_treino": round(float(y_tr.mean()), 4),
    }
    r = resultados[nome]
    print(f"\n[{nome}]")
    print(f"  ROC-AUC {r['roc_auc']:.3f} | PR-AUC {r['pr_auc']:.3f} "
          f"(ingênuo: {r['taxa_real_teste']:.3f})")
    print(f"  Com o limiar do treino ({limiar:.3f}): recall {r['recall']:.1%}, "
          f"precisão {r['precisao']:.1%}, alertas em {r['alertas_pct']:.1%} dos pedidos")
    print(f"  Probabilidade média prevista {r['prob_media_prevista']:.1%} "
          f"x taxa real no teste {r['taxa_real_teste']:.1%}")


todas = [c for c in base.columns if c not in colunas_aux]
avaliar("com_mes_compra", todas)
avaliar("sem_mes_compra", [c for c in todas if c != "mes_compra"])

# ── 4. Janelas móveis: várias "viagens ao futuro", não só uma ────────────────
# A última janela (3 meses) é calma e pode ser injusta. Aqui o modelo é treinado
# só com o que veio ANTES de cada janela e testado nela, em quatro momentos.
JANELAS = [("2017-11", "2018-01"), ("2018-02", "2018-03"),
           ("2018-04", "2018-05"), ("2018-06", "2018-08")]
rf_rapido = lambda: RandomForestClassifier(
    n_estimators=100, max_depth=8, class_weight="balanced", random_state=SEED, n_jobs=-1
)
janelas = []
print("\nJanelas móveis (treina antes da janela, testa nela):")
print(f"  {'Janela':<18}{'Treino':>8}{'Atraso':>8}{'ROC c/ mês':>12}{'ROC s/ mês':>12}{'PR-AUC':>8}{'Ingênuo':>9}")
sem_mes = [c for c in todas if c != "mes_compra"]
for ini, fim in JANELAS:
    passado = base[base["ano_mes"] < ini]
    janela = base[(base["ano_mes"] >= ini) & (base["ano_mes"] <= fim)]
    if janela["atrasou"].nunique() < 2 or len(passado) < 2000:
        continue
    com = rf_rapido().fit(passado[todas], passado["atrasou"])
    sem = rf_rapido().fit(passado[sem_mes], passado["atrasou"])
    p_com = com.predict_proba(janela[todas])[:, 1]
    p_sem = sem.predict_proba(janela[sem_mes])[:, 1]
    linha = {
        "janela": f"{ini} a {fim}", "pedidos_treino": len(passado),
        "taxa_atraso_janela": round(float(janela["atrasou"].mean()), 4),
        "roc_auc_com_mes": round(float(roc_auc_score(janela["atrasou"], p_com)), 4),
        "roc_auc_sem_mes": round(float(roc_auc_score(janela["atrasou"], p_sem)), 4),
        "pr_auc_com_mes": round(float(average_precision_score(janela["atrasou"], p_com)), 4),
    }
    janelas.append(linha)
    print(f"  {linha['janela']:<18}{len(passado):>8,}{linha['taxa_atraso_janela']:>8.1%}"
          f"{linha['roc_auc_com_mes']:>12.3f}{linha['roc_auc_sem_mes']:>12.3f}"
          f"{linha['pr_auc_com_mes']:>8.3f}{linha['taxa_atraso_janela']:>9.3f}")

os.makedirs(DOCS, exist_ok=True)
saida = {
    "janelas_moveis": janelas,
    "treino": {"inicio": f"{treino['data_compra'].min():%Y-%m-%d}",
               "fim": f"{treino['data_compra'].max():%Y-%m-%d}", "pedidos": len(treino)},
    "teste": {"inicio": f"{teste['data_compra'].min():%Y-%m-%d}",
              "fim": f"{teste['data_compra'].max():%Y-%m-%d}", "pedidos": len(teste)},
    "taxa_atraso_mensal": {m: round(float(v), 4) for m, v in mensal["mean"].items()},
    "resultados": resultados,
}
with open(os.path.join(DOCS, "validacao_temporal.json"), "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)
print("\nResultados salvos em docs/validacao_temporal.json")
