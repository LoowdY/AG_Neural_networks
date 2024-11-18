# Sistema Avançado de Previsão com GANN
### Universidade: *[Nome da Universidade]*
### Disciplina: Inteligência Artificial
### Professora: Apolyana Fonseca

### Alunos:
- João Renan Lopes
- Fellipe Torres
- Pedro Henrique Coimbra
- Carlos Egger Carvalho

## 1. Introdução

O Sistema Avançado de Previsão com GANN (Genetic Algorithm Neural Network) é uma implementação que combina algoritmos genéticos com redes neurais para previsão de séries temporais financeiras. O sistema utiliza dados históricos de ações para realizar previsões de preços futuros.

## 2. Arquitetura do Sistema

### 2.1 Componentes Principais

1. **Rede Neural Híbrida**
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
   - Camadas Dense para processamento final
   - Normalização e Dropout para regularização

2. **Algoritmo Genético**
   - População inicial de redes neurais
   - Seleção por elitismo
   - Crossover adaptativo
   - Mutação com taxa dinâmica

3. **Processamento de Dados**
   - Indicadores técnicos avançados
   - Normalização robusta
   - Criação de sequências temporais

## 3. Funcionalidades

### 3.1 Previsão de Preços
- Previsão de valores futuros
- Análise de múltiplos horizontes temporais
- Intervalos de confiança

### 3.2 Indicadores Técnicos
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Médias Móveis (SMA e EMA)
- ATR (Average True Range)
- Volume e OBV (On-Balance Volume)
- MFI (Money Flow Index)
- ROC (Rate of Change)

### 3.3 Análise de Performance
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- MSE (Mean Square Error)
- Correlação entre valores reais e previstos

## 4. Interface do Usuário

### 4.1 Parâmetros Configuráveis
- Tamanho da população (10-200)
- Número de gerações (10-300)
- Taxa de mutação (0.01-0.5)
- Taxa de crossover (0.1-1.0)
- Tamanho da elite (1-20)
- Janela temporal (5-30)
- Horizonte de previsão (1-4 dias)

### 4.2 Visualizações
- Gráfico de previsões vs valores reais
- Intervalos de confiança
- Evolução do fitness ao longo das gerações
- Métricas de performance em tempo real

## 5. Implementação Técnica

### 5.1 Tecnologias Utilizadas
- Python 3.x
- TensorFlow/Keras para redes neurais
- Pandas para manipulação de dados
- NumPy para computação numérica
- Streamlit para interface web
- Plotly para visualizações
- YFinance para dados financeiros

### 5.2 Principais Classes e Métodos

#### Classe GANNMelhorado
```python
# Métodos principais:
- __init__(): Inicialização com parâmetros configuráveis
- treinar(): Execução do algoritmo genético
- prever(): Realização de previsões
- criar_features_avancadas(): Geração de indicadores técnicos
- avaliar_modelo(): Cálculo de métricas de performance
```

## 6. Como Utilizar

1. **Instalação de Dependências**
```bash
pip install numpy pandas yfinance sklearn tensorflow ta plotly streamlit
```

2. **Execução do Sistema**
```bash
streamlit run nome_do_arquivo.py
```

3. **Configuração**
   - Inserir código da ação (ex: PETR4.SA)
   - Definir período de análise
   - Ajustar parâmetros do algoritmo genético
   - Iniciar análise

## 7. Considerações de Performance

- O sistema utiliza processamento paralelo para otimização
- Adaptação automática dos parâmetros genéticos durante o treinamento
- Escalabilidade para diferentes ativos financeiros
- Gestão de memória otimizada para grandes conjuntos de dados

## 8. Limitações e Melhorias Futuras

### 8.1 Limitações Atuais
- Processamento limitado ao hardware disponível
- Dependência da qualidade dos dados históricos
- Latência em ativos de alta volatilidade

### 8.2 Melhorias Propostas
- Implementação de GPU para processamento
- Adição de mais indicadores técnicos
- Otimização multi-objetivo
- Interface mais interativa
- Suporte a múltiplos ativos simultaneamente

## 9. Conclusão

O Sistema GANN representa uma abordagem avançada para previsão de séries temporais financeiras, combinando o poder dos algoritmos genéticos com redes neurais modernas. Sua implementação modular e interface intuitiva permitem tanto uso acadêmico quanto prático no mercado financeiro.

---
*Data da Documentação: 18 de Novembro de 2024*

