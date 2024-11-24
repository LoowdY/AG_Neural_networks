
# Sistema Avançado de Previsão com GANN

### Instituição: CESUPA - Centro Universitário do Pará
### Disciplina: Inteligência Artificial
### Professora: Polyana Fonseca Nascimento

### Equipe de Desenvolvimento:
1. João Renan S. Lopes
2. Carlos Egger
3. Pedro Coimbra 
4. Fellipe Torres

---

## Sumário

1. [Abstract](#1-abstract)
2. [Introdução](#2-introdução)
3. [Base Teórica](#3-base-teórica)
4. [Arquitetura do Sistema](#4-arquitetura-do-sistema)
5. [Implementação](#5-implementação)
6. [Interface do Usuário](#6-interface-do-usuário)
7. [Análise de Performance](#7-análise-de-performance)
8. [Guia de Instalação e Uso](#8-guia-de-instalação-e-uso)
9. [Resultados e Discussão](#9-resultados-e-discussão)
10. [Conclusões](#10-conclusões)
11. [Referências](#11-referências)

---

## 1. Abstract

Este projeto implementa um sistema híbrido avançado para previsão do mercado de ações, baseado no artigo "Hybrid artificial neural network and genetic algorithm model for stock market prediction" (DOI: 10.1007/s13198-021-01209-5). O sistema combina Redes Neurais Artificiais (RNA) com Algoritmos Genéticos (AG) para superar as limitações dos métodos tradicionais de previsão financeira.

## 2. Introdução

### 2.1 Contextualização
O mercado financeiro é caracterizado por sua alta volatilidade e não-linearidade, tornando a previsão de preços um desafio significativo. Métodos tradicionais e técnicas isoladas de IA frequentemente não capturam todas as nuances do mercado.

### 2.2 Objetivo
Desenvolver um sistema híbrido que combine:
- Redes Neurais Artificiais para aprendizado profundo
- Algoritmos Genéticos para otimização
- Indicadores técnicos avançados para análise de mercado

### 2.3 Justificativa
Conforme demonstrado no artigo base, a abordagem híbrida supera significativamente os métodos individuais em precisão de previsão, tanto em curto quanto em longo prazo.

## 3. Base Teórica

### 3.1 Redes Neurais Artificiais
- Arquitetura LSTM-GRU híbrida
- Camadas de normalização em lote
- Dropout para regularização
- Funções de ativação adaptativas

### 3.2 Algoritmos Genéticos
- Seleção por elitismo
- Crossover adaptativo
- Mutação com taxa dinâmica
- Avaliação de fitness baseada em erro de previsão

### 3.3 Indicadores Técnicos
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Médias Móveis (SMA/EMA)
- Indicadores de Volume (OBV, MFI)

## 4. Arquitetura do Sistema

### 4.1 Componentes Principais
1. **Módulo de Dados**
   - Coleta de dados via YFinance
   - Pré-processamento e normalização
   - Geração de features técnicas

2. **Módulo de RNA**
   - Camada LSTM (64 unidades)
   - Camada GRU (32 unidades)
   - Camadas Dense para processamento final
   - Normalização e Dropout

3. **Módulo de AG**
   - População de redes neurais
   - Operadores genéticos adaptados
   - Otimização multi-objetivo

4. **Interface do Usuário**
   - Dashboard Streamlit
   - Visualizações interativas
   - Controles de parâmetros

## 5. Implementação

### 5.1 Tecnologias Utilizadas
```python
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
```

### 5.2 Classes Principais

#### GANNMelhorado
```python
class GANNMelhorado:
    def __init__(self, tamanho_populacao, num_geracoes, ...):
        # Inicialização de parâmetros

    def treinar(self, X, y):
        # Implementação do treinamento

    def prever(self, X):
        # Implementação da previsão
```

## 6. Interface do Usuário

### 6.1 Configurações
- Seleção de ativo financeiro
- Período de análise
- Parâmetros do algoritmo genético
- Horizonte de previsão

### 6.2 Visualizações
- Gráfico de preços real vs previsto
- Intervalos de confiança
- Evolução do fitness
- Métricas de performance

## 7. Análise de Performance

### 7.1 Métricas
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- MSE (Mean Square Error)
- Correlação

## 8. Guia de Instalação e Uso

### 8.1 Requisitos
```bash
pip install -r requirements.txt
```

### 8.2 Execução
```bash
streamlit run gann_trading.py
```

## 9. Resultados e Discussão

### 9.1 Performance
- Precisão superior em períodos de alta volatilidade
- Adaptação eficiente a diferentes ativos
- Tempo de processamento otimizado

## 10. Conclusões

O sistema GANN demonstrou-se eficaz na previsão de séries temporais financeiras, corroborando os resultados do artigo base. A implementação adiciona funcionalidades práticas e uma interface moderna, tornando-o adequado tanto para uso acadêmico quanto profissional.

## 11. Referências

1. Artigo Base:
```
Título: Integration of genetic algorithm with artificial neural network for stock market forecasting
Journal: International Journal of System Assurance Engineering and Management
DOI: 10.1007/s13198-021-01209-5
```
