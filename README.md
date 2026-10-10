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
  "mensagem": "🟡 Risco MÉDIO de atraso (41.5%). Abaixo do limiar de alerta, mas vale acompanhar."
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

Treinado com **96.470 pedidos reais** do dataset Olist (e-commerce brasileiro, 2016–2018). Apenas **6,8% dos pedidos atrasam**, então o problema é bem desbalanceado e a acurácia sozinha engana.

| Métrica | Modelo | Referência |
|---|---|---|
| Recall (atrasos detectados) | **61,0%** | 0% (modelo que sempre diz "no prazo") |
| Precisão (alertas corretos) | **15,7%** | 6,8% (alertar ao acaso) |
| Acurácia | 75,2% | 93,2% (modelo que sempre diz "no prazo") |

**Como ler isso na prática.** A cada 1.000 pedidos, cerca de 68 atrasam. O modelo:

- detecta **~41 dos 68 atrasos** antes de acontecerem;
- gera **~264 alertas**, ou seja, a operação revisa ~26% dos pedidos em vez de 100%;
- tem precisão **2,3x maior** que escolher pedidos ao acaso.

**Por que priorizar o recall?** Na operação logística, um atraso não detectado custa mais (cliente insatisfeito, reclamação, reembolso) do que um alerta falso, que custa só uma checagem preventiva. A troca é consciente: a precisão é baixa e muitos alertas não viram atraso.

Por isso a API devolve a **probabilidade** e um **nível de risco**, não só sim/não:

| Nível | Probabilidade | Significado |
|---|---|---|
| ALTO | ≥ 50% | Alerta do modelo (`risco_atraso: true`). É neste limiar que valem as métricas acima |
| MÉDIO | 30% a 50% | Abaixo do limiar, mas vale acompanhar |
| BAIXO | < 30% | Risco baixo |

## Fatores mais importantes

| Feature | Importância |
|---|---|
| Mês da compra | 32,7% |
| Prazo prometido | 16,5% |
| Estado SP | 9,7% |
| Valor do frete | 9,1% |
| Estado RJ | 8,0% |

## Limitações conhecidas

- **Dependência do calendário:** o mês da compra é a feature dominante. No dataset Olist, os atrasos se concentram em períodos específicos (Black Friday de 2017, início de 2018). Parte do que o modelo aprendeu pode ser "quando" o pedido foi feito, e não um padrão que se repete em outros anos.
- **Validação:** o split atual é aleatório. O próximo passo é validar com **split temporal** (treinar no passado, testar no futuro), que simula o uso real em produção.
- **Importância das features:** calculada pelo método padrão do Random Forest, que tende a favorecer variáveis com muitas categorias. O plano é comparar com *permutation importance*.

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
