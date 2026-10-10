"""
Schema de entrada da API DeliveryRisk.
Define os campos que o cliente precisa enviar para obter a previsão.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


ESTADOS_BRASILEIROS = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO",
    "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
    "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"
]


class PedidoInput(BaseModel):
    """Dados do pedido disponíveis no momento da compra."""

    dia_semana_compra: int = Field(
        ..., ge=0, le=6,
        description="Dia da semana da compra (0=segunda, 6=domingo)"
    )
    prazo_prometido_dias: int = Field(
        ..., ge=1,
        description="Dias entre a compra e a data de entrega prometida"
    )
    qtd_itens: int = Field(
        ..., ge=1,
        description="Quantidade de itens no pedido"
    )
    valor_produtos: float = Field(
        ..., gt=0,
        description="Valor total dos produtos em reais"
    )
    valor_frete: float = Field(
        ..., ge=0,
        description="Valor do frete em reais"
    )
    peso_total_g: float = Field(
        ..., gt=0,
        description="Peso total dos produtos em gramas"
    )
    volume_total_cm3: float = Field(
        ..., gt=0,
        description="Volume total dos produtos em cm³"
    )
    estado_cliente: Literal[
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO",
        "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
        "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"
    ] = Field(
        ...,
        description="UF do cliente (sigla do estado)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dia_semana_compra": 4,
                "prazo_prometido_dias": 12,
                "qtd_itens": 2,
                "valor_produtos": 189.90,
                "valor_frete": 18.50,
                "peso_total_g": 800,
                "volume_total_cm3": 4000,
                "estado_cliente": "SP"
            }
        }
    )


class PrevisaoOutput(BaseModel):
    """Resultado da previsão de atraso."""

    risco_atraso: bool = Field(
        description="True quando há alerta de atraso (nível MÉDIO ou ALTO)"
    )
    probabilidade_atraso: float = Field(
        description="Probabilidade calibrada de atraso (0.0 a 1.0)"
    )
    nivel_risco: Literal["BAIXO", "MÉDIO", "ALTO"] = Field(
        description=(
            "BAIXO: abaixo do limiar de alerta. MÉDIO: alerta. "
            "ALTO: probabilidade de pelo menos o dobro da taxa média de atraso."
        )
    )
    mensagem: str = Field(description="Resumo legível do resultado")
