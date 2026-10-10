# DeliveryRisk API

**Modelo de previsão de risco de atraso de entregas em produção**  
*Python · FastAPI · scikit-learn · Docker · GCP Cloud Run*

**🔗 Demo ao vivo:** [Testar a API (Swagger)](https://deliveryrisk-api-889571326423.southamerica-east1.run.app/docs)

---

## Sobre o projeto

APIs de ML em produção são o que diferencia um projeto de portfólio de um produto real. Este projeto pega o modelo treinado no [Projeto 2 — Previsão de Atraso de Entrega](https://github.com/rafaelsimoesdev/previsao-atraso-entregas) e o coloca no ar com uma API REST documentada, containerizada com Docker e hospedada no Google Cloud Run.

**Problema:** Atrasos de entrega são um dos maiores problemas de satisfação no e-commerce. Identificar com antecedência quais pedidos têm risco de atraso permite que a operação logística tome ações preventivas antes que o cliente seja impactado.

**Solução:** Uma API que recebe os dados do pedido **no momento da compra** e retorna o risco de atraso em tempo real, com probabilidade e nível de risco.

---

## Demo

```bash
curl -X POST "https://deliveryrisk-api-889571326423.southamerica-east1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mes_compra": 11,
    "dia_semana_compra": 4,
    "prazo_prometido_dias": 12,
    "qtd_itens": 2,
    "valor_produtos": 189.90,
    "valor_frete": 18.50,
    "peso_total_g": 800,
    "volume_total_cm3": 4000,
    "estado_cliente": "SP"
  }'
```

**Resposta:**
```json
{
  "risco_atraso": false,
  "probabilidade_atraso": 0.415,
  "nivel_risco": "MÉDIO",
  "mensagem": "✅ Entrega dentro do prazo esperado. Probabilidade de atraso: 41.5% — nível MÉDIO."
}
```

Documentação interativa disponível em: https://deliveryrisk-api-889571326423.southamerica-east1.run.app/docs

---

## Stack

| Camada | Tecnologia |
|---|---|
| Modelo | Random Forest (scikit-learn) |
| API | FastAPI + Pydantic |
| Servidor | Uvicorn (ASGI) |
| Container | Docker |
| Deploy | Google Cloud Run |
| Linguagem | Python 3.11 |

---

## Estrutura do projeto

```
deliveryrisk-api/
├── app/
│   ├── main.py          # Endpoints da API (/, /health, /predict, /model/info)
│   └── schema.py        # Schemas de entrada e saída (Pydantic)
├── model/
│   └── model.pkl        # Random Forest serializado com joblib
├── train/
│   └── treinar_modelo.py  # Script para retreinar o modelo
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | Status da API |
| GET | `/health` | Health check (Cloud Run) |
| POST | `/predict` | Previsão de risco de atraso |
| GET | `/model/info` | Informações e métricas do modelo |
| GET | `/docs` | Documentação Swagger (automática) |

---

## Métricas do modelo

O modelo foi treinado com **96.470 pedidos reais** do dataset Olist (e-commerce brasileiro, 2016-2018):

| Métrica | Valor |
|---|---|
| Acurácia | 75,2% |
| Precisão | 15,7% |
| **Recall** | **61,0%** |

**Por que o recall é a métrica principal?** Com apenas 6,8% de atrasos, um modelo que erra para o lado conservador é melhor do que um que deixa atrasos passarem. O modelo detecta **6 de cada 10 atrasos reais** antes da entrega.

### Fatores mais importantes

| Feature | Importância |
|---|---|
| Mês da compra | 32,7% |
| Prazo prometido | 16,5% |
| Estado SP | 9,7% |
| Valor do frete | 9,1% |
| Estado RJ | 8,0% |

O **mês da compra** é o fator dominante — confirma que períodos de pico (Black Friday, início de 2018) são o maior risco operacional.

---

## Como rodar localmente

### Com Docker (recomendado)

```bash
# Build
docker build -t deliveryrisk-api .

# Run
docker run -p 8080:8080 deliveryrisk-api
```

Acesse: `http://localhost:8080/docs`

### Sem Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

---

## Como retreinar o modelo

```bash
# Coloque base_modelo.csv na raiz do projeto, depois:
python train/treinar_modelo.py
```

O script salva o novo `model.pkl` em `model/` automaticamente.

---

## Deploy no GCP Cloud Run

```bash
# Autenticar
gcloud auth login

# Build e push para o Container Registry
gcloud builds submit --tag gcr.io/SEU_PROJETO/deliveryrisk-api

# Deploy
gcloud run deploy deliveryrisk-api \
  --image gcr.io/SEU_PROJETO/deliveryrisk-api \
  --platform managed \
  --region southamerica-east1 \
  --allow-unauthenticated
```

---

## Contexto

Este projeto faz parte de um portfólio de ML com foco em logística e e-commerce:

- **Projeto 2 — [Previsão de Atraso de Entrega](https://github.com/rafaelsimoesdev/previsao-atraso-entregas):** EDA, pré-processamento e treinamento dos modelos
- **Projeto 3 — DeliveryRisk API (este repositório):** Deploy do modelo em produção com FastAPI + Docker + GCP

**Autor:** Rafael Simões — [LinkedIn](https://linkedin.com/in/rafael-simões-556979239) · [GitHub](https://github.com/rafaelsimoesdev)  
**Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle
