"""
Projeto 3 — DeliveryRisk API
Etapa 1: Retreinar o Random Forest com a base completa e serializar o modelo.

Este script é a origem do model.pkl usado pela API FastAPI.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import joblib
import os

# ── Caminhos (relativos ao próprio script, funcionam de qualquer pasta) ──────
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(RAIZ, "base_modelo.csv")  # coloque base_modelo.csv na raiz do projeto
OUTPUT_PATH = os.path.join(RAIZ, "model", "model.pkl")

# ── Carregar base ────────────────────────────────────────────────────────────
base = pd.read_csv(CSV_PATH)

print(f"Base carregada: {base.shape[0]:,} pedidos | {base.shape[1]-1} features")
print(f"Taxa de atraso: {base['atrasou'].mean()*100:.1f}%")
print(f"Features: {list(base.drop(columns=['atrasou']).columns)}")

# ── Separar features e target ────────────────────────────────────────────────
X = base.drop(columns=["atrasou"])
y = base["atrasou"]

FEATURE_NAMES = list(X.columns)

# ── Validação rápida (80/20) para confirmar métricas ────────────────────────
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Treinar modelo ───────────────────────────────────────────────────────────
print("\nTreinando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_treino, y_treino)

# ── Métricas de validação ────────────────────────────────────────────────────
y_pred = rf.predict(X_teste)
acc   = (y_pred == y_teste).mean()
prec  = precision_score(y_teste, y_pred)
rec   = recall_score(y_teste, y_pred)

print(f"\n{'='*45}")
print(f"{'Métrica':<20} {'Valor':>10}")
print(f"{'-'*45}")
print(f"{'Acurácia':<20} {acc*100:>9.1f}%")
print(f"{'Precisão':<20} {prec*100:>9.1f}%")
print(f"{'Recall':<20} {rec*100:>9.1f}%")
print(f"{'='*45}")

# ── Salvar modelo + feature names ────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

modelo_payload = {
    "model": rf,
    "feature_names": FEATURE_NAMES
}

joblib.dump(modelo_payload, OUTPUT_PATH)

print(f"\nModelo salvo em: {OUTPUT_PATH}")
print(f"Features salvas: {len(FEATURE_NAMES)}")

# ── Teste rápido de carga ────────────────────────────────────────────────────
loaded = joblib.load(OUTPUT_PATH)
print(f"Teste de carga: OK — {len(loaded['feature_names'])} features, modelo tipo {type(loaded['model']).__name__}")
