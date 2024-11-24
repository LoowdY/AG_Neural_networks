import os
import threading
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, GRU, Input
from tensorflow.keras.optimizers import Adam
import ta
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import warnings
from typing import Tuple, Callable, Optional
import logging
import queue
import time
import random  # Importado para estratégias de seleção no AG

# Suprimir avisos do TensorFlow e outros
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configurações para otimizar o TensorFlow na CPU
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
tf.config.optimizer.set_jit(True)  # Ativar XLA (Acelerador Linear)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': False})

# Configuração de logging
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    """Handler que envia logs para uma fila."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
queue_handler = QueueHandler(log_queue)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

def criar_modelo_hibrido(input_shape):
    """
    Cria e compila um modelo híbrido LSTM-GRU-Dense.
    
    Parâmetros:
    - input_shape: Tupla que define a forma de entrada do modelo.
    
    Retorna:
    - Modelo compilado.
    """
    modelo = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(16, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(8),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    modelo.compile(optimizer=optimizer, loss='huber')
    return modelo

class GANNMelhorado:
    """
    Classe que implementa o Algoritmo Genético Melhorado para otimizar os pesos de uma rede neural.
    """
    def __init__(
            self,
            tamanho_populacao: int = 10,
            num_geracoes: int = 10,
            taxa_mutacao: float = 0.1,
            taxa_crossover: float = 0.7,
            tamanho_elite: int = 2,
            janela_temporal: int = 5,
            estrategia_selecao: str = 'roulette'  # 'roulette' ou 'tournament'
    ):
        """
        Inicializa os parâmetros do Algoritmo Genético.
        
        Parâmetros:
        - tamanho_populacao: Número de indivíduos na população.
        - num_geracoes: Número de gerações para evoluir.
        - taxa_mutacao: Taxa de mutação aplicada nos indivíduos.
        - taxa_crossover: Taxa de crossover entre indivíduos.
        - tamanho_elite: Número de indivíduos elitistas a serem mantidos.
        - janela_temporal: Tamanho da janela temporal para as sequências.
        - estrategia_selecao: Estratégia de seleção ('roulette' ou 'tournament').
        """
        self.tamanho_populacao = tamanho_populacao
        self.num_geracoes = num_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.tamanho_elite = tamanho_elite
        self.janela_temporal = janela_temporal
        self.estrategia_selecao = estrategia_selecao

        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

        self.historico_fitness = []
        self.fitness_history = []  # Para plotar o gráfico de fitness
        self.melhor_modelo = None

    def crossover_adaptativo(self, pai1, pai2):
        """
        Realiza crossover adaptativo entre dois conjuntos de pesos.
        
        Parâmetros:
        - pai1: Pesos do primeiro pai.
        - pai2: Pesos do segundo pai.
        
        Retorna:
        - Filhos resultantes do crossover.
        """
        filho = []
        for p1, p2 in zip(pai1, pai2):
            if random.random() < self.taxa_crossover:
                # Crossover aritmético adaptativo
                alpha = random.random()
                peso_filho = alpha * p1 + (1 - alpha) * p2
            else:
                # Seleciona um dos pais aleatoriamente
                peso_filho = p1 if random.random() < 0.5 else p2
            filho.append(peso_filho)
        return filho

    def mutacao_adaptativa(self, individuo, geracao):
        """
        Realiza mutação adaptativa nos pesos.
        
        Parâmetros:
        - individuo: Conjunto de pesos do indivíduo.
        - geracao: Número atual da geração.
        
        Retorna:
        - Indivíduo mutado.
        """
        # Taxa de mutação adaptativa que diminui com as gerações
        taxa_atual = self.taxa_mutacao * (1 - geracao / self.num_geracoes)

        for i in range(len(individuo)):
            if random.random() < taxa_atual:
                # Magnitude da mutação diminui com as gerações
                magnitude = random.gauss(0, 0.05 * (1 - geracao / self.num_geracoes))
                individuo[i] = individuo[i] + magnitude * np.random.randn(*individuo[i].shape)
        return individuo

    def criar_features_avancadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features técnicas avançadas garantindo que as entradas sejam Series unidimensionais.
        
        Parâmetros:
        - df: DataFrame contendo os dados de preços.
        
        Retorna:
        - DataFrame com as novas features.
        """
        try:
            df = df.copy()

            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            close = close.squeeze() if isinstance(close, pd.DataFrame) else close
            high = high.squeeze() if isinstance(high, pd.DataFrame) else high
            low = low.squeeze() if isinstance(low, pd.DataFrame) else low
            volume = volume.squeeze() if isinstance(volume, pd.DataFrame) else volume

            close = pd.Series(close) if not isinstance(close, pd.Series) else close
            high = pd.Series(high) if not isinstance(high, pd.Series) else high
            low = pd.Series(low) if not isinstance(low, pd.Series) else low
            volume = pd.Series(volume) if not isinstance(volume, pd.Series) else volume

            # RSI
            rsi = ta.momentum.RSIIndicator(close)
            df['RSI'] = rsi.rsi()

            # MACD
            macd = ta.trend.MACD(close)
            df['MACD_diff'] = macd.macd_diff()

            # ATR
            atr = ta.volatility.AverageTrueRange(high, low, close)
            df['ATR'] = atr.average_true_range()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = df['BB_upper'] - df['BB_lower']

            # Médias Móveis
            for periodo in [5, 10]:
                df[f'SMA_{periodo}'] = ta.trend.sma_indicator(close, window=periodo)
                df[f'EMA_{periodo}'] = ta.trend.ema_indicator(close, window=periodo)

            # Volume
            obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
            df['OBV'] = obv.on_balance_volume()
            df['Volume_SMA'] = ta.trend.sma_indicator(volume, window=20)
            df['Volume_ratio'] = volume / df['Volume_SMA']

            # Money Flow Index
            mfi = ta.volume.MFIIndicator(high, low, close, volume)
            df['MFI'] = mfi.money_flow_index()

            # Rate of Change
            roc = ta.momentum.ROCIndicator(close)
            df['ROC'] = roc.roc()

            # Log Returns e Volatilidade
            df['Log_return'] = np.log(close).diff()
            df['Volatility'] = df['Log_return'].rolling(window=20).std()

            return df.dropna()

        except Exception as e:
            logger.error(f"Erro ao criar indicadores técnicos: {str(e)}")
            raise

    def criar_sequencias_temporais(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências temporais para análise temporal.
        
        Parâmetros:
        - X: Dados de entrada.
        - y: Dados de saída.
        
        Retorna:
        - Tupla contendo X sequencial e y sequencial.
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - self.janela_temporal):
            X_seq.append(X[i:(i + self.janela_temporal)])
            y_seq.append(y[i + self.janela_temporal])
        return np.array(X_seq), np.array(y_seq)

    def preparar_dados(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara os dados para treinamento.
        
        Parâmetros:
        - df: DataFrame contendo os dados de preços.
        
        Retorna:
        - Tupla contendo X e y preparados.
        """
        try:
            df = self.criar_features_avancadas(df)

            features = df.drop(columns=['Close']).columns
            X = df[features].values
            y = df['Close'].values

            X = self.scaler_x.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

            return X, y

        except Exception as e:
            logger.error(f"Erro na preparação dos dados: {str(e)}")
            raise

    def selecionar_pais(self, populacao, fitness_scores):
        """
        Seleciona os pais da população com base na estratégia de seleção.
        
        Parâmetros:
        - populacao: Lista de indivíduos (pesos).
        - fitness_scores: Lista de scores de fitness correspondentes.
        
        Retorna:
        - Lista de pais selecionados.
        """
        if self.estrategia_selecao == 'roulette':
            return self.selecao_roleta(populacao, fitness_scores)
        elif self.estrategia_selecao == 'tournament':
            return self.selecao_torneio(populacao, fitness_scores)
        else:
            raise ValueError("Estratégia de seleção inválida.")

    def selecao_roleta(self, populacao, fitness_scores):
        """
        Implementa a seleção por roleta.
        
        Parâmetros:
        - populacao: Lista de indivíduos.
        - fitness_scores: Lista de scores de fitness.
        
        Retorna:
        - Lista de indivíduos selecionados.
        """
        # Utiliza random.choices com pesos para seleção por roleta
        pais = random.choices(populacao, weights=fitness_scores, k=self.tamanho_elite)
        return pais

    def selecao_torneio(self, populacao, fitness_scores, torneio_size=3):
        """
        Implementa a seleção por torneio.
        
        Parâmetros:
        - populacao: Lista de indivíduos.
        - fitness_scores: Lista de scores de fitness.
        - torneio_size: Número de indivíduos em cada torneio.
        
        Retorna:
        - Lista de indivíduos selecionados.
        """
        pais = []
        for _ in range(self.tamanho_elite):
            # Seleciona aleatoriamente 'torneio_size' indivíduos
            participantes = random.sample(list(zip(populacao, fitness_scores)), torneio_size)
            # Seleciona o indivíduo com maior fitness
            vencedor = max(participantes, key=lambda x: x[1])[0]
            pais.append(vencedor)
        return pais

    def treinar(self, X, y, callback: Optional[Callable[[int, float, float, float], None]] = None):
        """
        Treina o modelo usando otimização por Algoritmo Genético.
        
        Parâmetros:
        - X: Dados de entrada.
        - y: Dados de saída.
        - callback: Função de callback para atualizar a interface após cada geração.
        """
        # Cria sequências temporais
        X_seq, y_seq = self.criar_sequencias_temporais(X, y)
        split = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        input_shape = (self.janela_temporal, X.shape[-1])
        populacao = [criar_modelo_hibrido(input_shape).get_weights() for _ in range(self.tamanho_populacao)]

        start_time = time.time()

        for geracao in range(self.num_geracoes):
            logger.info(f"Iniciando Geração {geracao + 1}/{self.num_geracoes}")

            fitness_scores = []
            for idx, pesos in enumerate(populacao):
                try:
                    modelo = criar_modelo_hibrido(input_shape)
                    modelo.set_weights(pesos)
                    loss = modelo.evaluate(X_val, y_val, verbose=0)
                    fitness_scores.append(loss)
                    logger.info(f"Geração {geracao + 1}, Indivíduo {idx + 1}: Loss={loss:.4f}")
                except Exception as e:
                    logger.error(f"Erro ao avaliar indivíduo {idx + 1} na geração {geracao + 1}: {str(e)}")
                    fitness_scores.append(float('inf'))

            # Converter loss em fitness (quanto menor o loss, maior o fitness)
            fitness_scores_mape = [1 / (1 + loss) if loss != float('inf') else 0 for loss in fitness_scores]

            # Selecionar elite com base na estratégia de seleção
            elite = self.selecionar_pais(populacao, fitness_scores_mape)
            nova_populacao = elite.copy()

            logger.info(f"Selecionando elite: {len(elite)} indivíduos")

            # Armazenar fitness da geração para plotar
            melhor_fitness = max(fitness_scores_mape)
            self.fitness_history.append(melhor_fitness)

            # Crossover e Mutação para criar novos indivíduos
            while len(nova_populacao) < self.tamanho_populacao:
                # Seleciona dois pais aleatoriamente da elite
                pais = random.sample(elite, 2) if len(elite) >= 2 else [elite[0], elite[0]]
                filho = self.crossover_adaptativo(pais[0], pais[1])
                filho = self.mutacao_adaptativa(filho, geracao)
                nova_populacao.append(filho)
                logger.info(f"Criando novo indivíduo via crossover e mutação")

            populacao = nova_populacao[:self.tamanho_populacao]

            logger.info(f"Geração {geracao + 1}: Melhor Fitness={melhor_fitness:.4f}")

            # Atualizar o melhor modelo se o fitness atual for melhor
            if not self.melhor_modelo or melhor_fitness > max(self.historico_fitness or [0]):
                self.melhor_modelo = criar_modelo_hibrido(input_shape)
                self.melhor_modelo.set_weights(populacao[np.argmax(fitness_scores_mape)])
                logger.info(f"Atualizando melhor modelo com o indivíduo da geração {geracao + 1}")

            self.historico_fitness.append(melhor_fitness)

            # Chamar o callback com informações para atualizar a interface
            if callback:
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / (geracao + 1)) * self.num_geracoes
                remaining_time = estimated_total_time - elapsed_time
                callback(geracao + 1, melhor_fitness, np.mean(fitness_scores_mape), remaining_time)

            # Limpar a memória para evitar vazamento
            tf.keras.backend.clear_session()

        logger.info("Treinamento concluído.")

    def prever(self, X):
        """
        Faz previsões usando o melhor modelo.
        
        Parâmetros:
        - X: Dados de entrada.
        
        Retorna:
        - Previsões invertidas da escala.
        """
        X_seq = np.array([X[i:i + self.janela_temporal] for i in range(len(X) - self.janela_temporal)])
        previsoes_norm = self.melhor_modelo.predict(X_seq)
        return self.scaler_y.inverse_transform(previsoes_norm)

    def prever_n_dias(self, X, n_dias):
        """
        Faz previsões para n dias à frente.
        
        Parâmetros:
        - X: Dados de entrada.
        - n_dias: Número de dias para prever.
        
        Retorna:
        - Previsões invertidas da escala.
        """
        previsoes = []
        X_atual = X[-self.janela_temporal:].copy()

        for _ in range(n_dias):
            X_seq = X_atual.reshape(1, self.janela_temporal, -1)
            previsao = self.melhor_modelo.predict(X_seq, verbose=0)[0]
            previsoes.append(previsao)
            X_atual = np.roll(X_atual, -1, axis=0)
            X_atual[-1] = previsao

        return self.scaler_y.inverse_transform(np.array(previsoes).reshape(-1, 1))

    def avaliar_modelo(self, y_true, y_pred):
        """
        Avalia o desempenho do modelo.
        
        Parâmetros:
        - y_true: Valores reais.
        - y_pred: Valores previstos.
        
        Retorna:
        - Dicionário com métricas de avaliação.
        """
        # Evitar divisão por zero adicionando um epsilon
        epsilon = 1e-10
        y_true = np.where(y_true == 0, epsilon, y_true)
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return {
            'MAPE': mape,
            'MSE': mse,
            'RMSE': rmse
        }

    def plotar_resultados(self, y_true, y_pred, titulo="Resultados de Previsão"):
        """
        Plota os resultados da previsão com intervalos de confiança.
        
        Parâmetros:
        - y_true: Valores reais.
        - y_pred: Valores previstos.
        - titulo: Título do gráfico.
        
        Retorna:
        - Figura do Matplotlib.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y_true, label='Real', color='blue')
        ax.plot(y_pred, label='Previsto', color='red')
        erro = y_true - y_pred
        std_erro = np.std(erro)
        ax.fill_between(range(len(y_pred)), y_pred + 2 * std_erro, y_pred - 2 * std_erro, color='red', alpha=0.1, label='Intervalo de Confiança')
        ax.set_title(titulo)
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Preço de Fechamento')
        ax.legend()
        ax.grid(True)
        return fig

    def plotar_fitness(self, fitness_history, titulo="Evolução do Fitness"):
        """
        Plota a evolução do fitness ao longo das gerações.
        
        Parâmetros:
        - fitness_history: Lista de valores de fitness ao longo das gerações.
        - titulo: Título do gráfico.
        
        Retorna:
        - Figura do Matplotlib.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(fitness_history) + 1), fitness_history, marker='o', color='green')
        ax.set_title(titulo)
        ax.set_xlabel('Geração')
        ax.set_ylabel('Fitness')
        ax.grid(True)
        return fig

    def gerar_relatorio(self, y_true, y_pred, periodo=""):
        """
        Gera um relatório detalhado de performance.
        
        Parâmetros:
        - y_true: Valores reais.
        - y_pred: Valores previstos.
        - periodo: Período de análise (opcional).
        
        Retorna:
        - String contendo o relatório.
        """
        metricas = self.avaliar_modelo(y_true, y_pred)
        relatorio = f"""
RELATÓRIO DE PERFORMANCE {periodo}
===============================
Métricas Principais:
- MAPE: {metricas['MAPE']:.2f}%
- RMSE: {metricas['RMSE']:.4f}
- MSE: {metricas['MSE']:.4f}

Análise de Erro:
- Erro Médio: {np.mean(y_true - y_pred):.4f}
- Erro Máximo: {np.max(np.abs(y_true - y_pred)):.4f}
- Desvio Padrão do Erro: {np.std(y_true - y_pred):.4f}

Estatísticas de Previsão:
- Média Real: {np.mean(y_true):.4f}
- Média Prevista: {np.mean(y_pred):.4f}
- Correlação: {np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]:.4f}
"""
        return relatorio

class AplicacaoTkinter:
    """
    Classe que implementa a interface gráfica utilizando Tkinter.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Avançado de Previsão com GANN")
        self.root.geometry("1200x800")
        self.modelo = None
        self.thread = None
        self.previsoes_futuras = None

        self.create_widgets()
        self.root.after(100, self.process_log_queue)

    def create_widgets(self):
        """
        Cria os widgets da interface gráfica.
        """
        # Criar Notebook para abas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Abas
        self.aba_config = ttk.Frame(self.notebook)
        self.aba_previsao = ttk.Frame(self.notebook)
        self.aba_fitness = ttk.Frame(self.notebook)
        self.aba_estatisticas = ttk.Frame(self.notebook)

        self.notebook.add(self.aba_config, text='Configurações e Início')
        self.notebook.add(self.aba_previsao, text='Gráficos de Previsão')
        self.notebook.add(self.aba_fitness, text='Gráficos de Fitness')
        self.notebook.add(self.aba_estatisticas, text='Análise Estatística')

        # --- Aba de Configurações e Início ---
        config_frame = ttk.LabelFrame(self.aba_config, text="Configurações")
        config_frame.pack(fill="x", padx=10, pady=5)

        # Ticker
        ttk.Label(config_frame, text="Código da Ação:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.ticker_entry = ttk.Entry(config_frame, width=20)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.ticker_entry.insert(0, "PETR4.SA")  # Valor padrão

        # Data Inicial
        ttk.Label(config_frame, text="Data Inicial (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.data_inicio_entry = ttk.Entry(config_frame, width=20)
        self.data_inicio_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.data_inicio_entry.insert(0, "2020-01-01")  # Valor padrão

        # Data Final
        ttk.Label(config_frame, text="Data Final (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.data_fim_entry = ttk.Entry(config_frame, width=20)
        self.data_fim_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.data_fim_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))  # Valor padrão

        # Frame para parâmetros do algoritmo genético
        param_frame = ttk.LabelFrame(config_frame, text="Parâmetros do Algoritmo Genético")
        param_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Tamanho da População
        ttk.Label(param_frame, text="Tamanho da População:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.pop_size = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=5, to=50, textvariable=self.pop_size, width=5).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Número de Gerações
        ttk.Label(param_frame, text="Número de Gerações:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.num_gen = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=5, to=100, textvariable=self.num_gen, width=5).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Taxa de Mutação
        ttk.Label(param_frame, text="Taxa de Mutação:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.mut_rate = tk.DoubleVar(value=0.1)
        ttk.Spinbox(param_frame, from_=0.01, to=0.5, increment=0.01, textvariable=self.mut_rate, width=5).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Taxa de Crossover
        ttk.Label(param_frame, text="Taxa de Crossover:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.crossover_rate = tk.DoubleVar(value=0.7)
        ttk.Spinbox(param_frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.crossover_rate, width=5).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Tamanho da Elite
        ttk.Label(param_frame, text="Tamanho da Elite:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.elite_size = tk.IntVar(value=2)
        ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.elite_size, width=5).grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Janela Temporal
        ttk.Label(param_frame, text="Janela Temporal:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.janela_temporal = tk.IntVar(value=5)
        ttk.Spinbox(param_frame, from_=3, to=20, textvariable=self.janela_temporal, width=5).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Estratégia de Seleção
        ttk.Label(param_frame, text="Estratégia de Seleção:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.estrategia_selecao = tk.StringVar(value='roulette')
        ttk.Combobox(param_frame, textvariable=self.estrategia_selecao, values=['roulette', 'tournament'], state="readonly", width=12).grid(row=6, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(param_frame, text="(roulette/tournament)").grid(row=6, column=2, padx=5, pady=5, sticky="w")

        # Frame para parâmetros de previsão
        previsao_frame = ttk.LabelFrame(config_frame, text="Parâmetros de Previsão")
        previsao_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Dias para Previsão
        ttk.Label(previsao_frame, text="Dias para Previsão:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.dias_previsao = tk.IntVar(value=1)
        dias_options = [1, 2, 3, 4]
        ttk.Combobox(previsao_frame, textvariable=self.dias_previsao, values=dias_options, state="readonly", width=5).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Botão para iniciar análise
        self.iniciar_button = ttk.Button(config_frame, text="Iniciar Análise", command=self.iniciar_analise)
        self.iniciar_button.grid(row=7, column=0, columnspan=2, pady=10)

        # Barra de progresso
        self.progress = ttk.Progressbar(config_frame, orient='horizontal', mode='determinate', length=400)
        self.progress.grid(row=8, column=0, columnspan=2, pady=10)

        # Estimativa de Tempo Restante
        self.tempo_restante_label = ttk.Label(config_frame, text="Tempo Restante: N/A")
        self.tempo_restante_label.grid(row=9, column=0, columnspan=2, pady=5)

        # --- Aba de Logs ---
        log_frame = ttk.LabelFrame(self.aba_config, text="Logs do Treinamento")
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_frame, height=15, wrap='word', state='disabled')
        self.log_text.pack(fill="both", expand=True)

        # --- Aba de Gráficos de Previsão ---
        grafico_previsao_frame = ttk.LabelFrame(self.aba_previsao, text="Gráfico de Previsão")
        grafico_previsao_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig_previsao, self.ax_previsao = plt.subplots(figsize=(8, 4))
        self.canvas_previsao = FigureCanvasTkAgg(self.fig_previsao, master=grafico_previsao_frame)
        self.canvas_previsao.draw()
        self.canvas_previsao.get_tk_widget().pack(fill='both', expand=True)

        # --- Aba de Gráficos de Fitness ---
        grafico_fitness_frame = ttk.LabelFrame(self.aba_fitness, text="Gráfico de Fitness")
        grafico_fitness_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig_fitness, self.ax_fitness = plt.subplots(figsize=(8, 4))
        self.canvas_fitness = FigureCanvasTkAgg(self.fig_fitness, master=grafico_fitness_frame)
        self.canvas_fitness.draw()
        self.canvas_fitness.get_tk_widget().pack(fill='both', expand=True)

        # --- Aba de Análise Estatística ---
        estatisticas_frame = ttk.LabelFrame(self.aba_estatisticas, text="Análise Estatística")
        estatisticas_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Texto do relatório
        self.relatorio_text = tk.Text(estatisticas_frame, height=15, wrap='word', state='disabled')
        self.relatorio_text.pack(fill="both", expand=True)

        # Botão para baixar CSV
        self.baixar_button = ttk.Button(estatisticas_frame, text="Baixar Previsões em CSV", command=self.baixar_csv)
        self.baixar_button.pack(pady=10)

    def iniciar_analise(self):
        """
        Inicia a análise de previsão acionando uma thread separada.
        """
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("Aviso", "Análise já está em andamento.")
            return

        ticker = self.ticker_entry.get().strip()
        data_inicio = self.data_inicio_entry.get().strip()
        data_fim = self.data_fim_entry.get().strip()

        # Validação das datas
        try:
            pd.to_datetime(data_inicio)
            pd.to_datetime(data_fim)
        except ValueError:
            messagebox.showerror("Erro", "Datas inválidas. Use o formato YYYY-MM-DD.")
            return

        if not ticker:
            messagebox.showerror("Erro", "Código da Ação não pode estar vazio.")
            return

        # Obter parâmetros do usuário
        pop_size = self.pop_size.get()
        num_gen = self.num_gen.get()
        mut_rate = self.mut_rate.get()
        crossover_rate = self.crossover_rate.get()
        elite_size = self.elite_size.get()
        janela_temporal = self.janela_temporal.get()
        estrategia_selecao = self.estrategia_selecao.get()
        dias_previsao = self.dias_previsao.get()

        # Resetar métricas e relatório
        self.relatorio_text.config(state='normal')
        self.relatorio_text.delete('1.0', tk.END)
        self.relatorio_text.config(state='disabled')

        # Limpar gráficos de previsão
        self.fig_previsao.clf()
        self.ax_previsao = self.fig_previsao.add_subplot(111)
        self.canvas_previsao.draw()

        # Limpar gráficos de fitness
        self.fig_fitness.clf()
        self.ax_fitness = self.fig_fitness.add_subplot(111)
        self.canvas_fitness.draw()

        # Resetar barra de progresso e tempo restante
        self.progress['value'] = 0
        self.tempo_restante_label.config(text="Tempo Restante: N/A")
        self.root.update_idletasks()

        # Iniciar thread de análise
        self.thread = threading.Thread(target=self.analise, args=(
            ticker, data_inicio, data_fim, pop_size, num_gen,
            mut_rate, crossover_rate, elite_size, janela_temporal, dias_previsao, estrategia_selecao
        ), daemon=True)
        self.thread.start()

    def analise(self, ticker, data_inicio, data_fim, pop_size, num_gen,
               mut_rate, crossover_rate, elite_size, janela_temporal, dias_previsao, estrategia_selecao):
        """
        Executa a análise de previsão em uma thread separada.
        """
        try:
            # Baixar dados
            logger.info(f"Baixando dados para {ticker} de {data_inicio} a {data_fim}...")
            df = yf.download(ticker, start=data_inicio, end=data_fim)
            if df.empty:
                logger.error("Não foi possível obter dados para o ticker informado.")
                messagebox.showerror("Erro", "Não foi possível obter dados para o ticker informado.")
                return

            required_columns = ['Close', 'High', 'Low', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Coluna '{col}' ausente nos dados baixados.")
                    messagebox.showerror("Erro", f"Coluna '{col}' ausente nos dados baixados.")
                    return

            # Preparar dados
            logger.info("Preparando dados para treinamento...")
            self.modelo = GANNMelhorado(
                tamanho_populacao=pop_size,
                num_geracoes=num_gen,
                taxa_mutacao=mut_rate,
                taxa_crossover=crossover_rate,
                tamanho_elite=elite_size,
                janela_temporal=janela_temporal,
                estrategia_selecao=estrategia_selecao
            )

            X, y = self.modelo.preparar_dados(df)

            # Atualizar barra de progresso máxima
            self.progress['maximum'] = num_gen

            # Callback para atualizar a interface após cada geração
            def callback(geracao, melhor_fitness, fitness_medio, remaining_time):
                logger.info(f"Geração {geracao}/{num_gen} concluída.")
                self.progress['value'] = geracao
                self.tempo_restante_label.config(text=f"Tempo Restante: {remaining_time:.2f} segundos")
                self.root.update_idletasks()

                # Atualizar gráfico de fitness dinamicamente
                self.fig_fitness.clf()
                self.fig_fitness = self.modelo.plotar_fitness(self.modelo.fitness_history, titulo="Evolução do Fitness")
                self.canvas_fitness = FigureCanvasTkAgg(self.fig_fitness, master=self.aba_fitness)
                self.canvas_fitness.draw()
                self.canvas_fitness.get_tk_widget().pack(fill='both', expand=True)

            # Treinar modelo
            logger.info("Iniciando treinamento do modelo...")
            self.modelo.treinar(X, y, callback=callback)

            # Realizar previsões
            logger.info("Realizando previsões com o melhor modelo...")
            previsoes = self.modelo.prever(X)
            previsoes_futuras = self.modelo.prever_n_dias(X, dias_previsao)

            # Avaliar modelo
            logger.info("Avaliando o desempenho do modelo...")
            # Inversão correta de escala para y_true
            y_true_evaluation_scaled = y[-len(previsoes):]
            y_true_evaluation = self.modelo.scaler_y.inverse_transform(y_true_evaluation_scaled.reshape(-1,1)).flatten()
            y_pred_evaluation = previsoes.flatten()
            metricas = self.modelo.avaliar_modelo(y_true_evaluation, y_pred_evaluation)

            # Atualizar métricas na interface
            relatorio = self.modelo.gerar_relatorio(y_true_evaluation, y_pred_evaluation, periodo="")
            self.relatorio_text.config(state='normal')
            self.relatorio_text.delete('1.0', tk.END)
            self.relatorio_text.insert(tk.END, relatorio)
            self.relatorio_text.config(state='disabled')

            # Plotar gráficos de previsão dinamicamente
            self.fig_previsao.clf()
            self.fig_previsao = self.modelo.plotar_resultados(y_true_evaluation, y_pred_evaluation, titulo="Comparação Real vs Previsto")
            self.canvas_previsao = FigureCanvasTkAgg(self.fig_previsao, master=self.aba_previsao)
            self.canvas_previsao.draw()
            self.canvas_previsao.get_tk_widget().pack(fill='both', expand=True)

            # Informar ao usuário que o treinamento foi concluído
            logger.info("Treinamento e previsão concluídos com sucesso.")
            messagebox.showinfo("Concluído", "Treinamento e previsão concluídos com sucesso.")

            # Armazenar previsões futuras para download
            self.previsoes_futuras = previsoes_futuras.flatten()

        except Exception as e:
            logger.error(f"Erro durante a execução: {str(e)}")
            messagebox.showerror("Erro", f"Erro durante a execução: {str(e)}")

    def process_log_queue(self):
        """
        Processa a fila de logs e atualiza o widget de log.
        """
        while not log_queue.empty():
            try:
                record = log_queue.get_nowait()
                self.log_text.config(state='normal')
                self.log_text.insert(tk.END, record + '\n')
                self.log_text.see(tk.END)
                self.log_text.config(state='disabled')
            except queue.Empty:
                pass
        self.root.after(100, self.process_log_queue)

    def baixar_csv(self):
        """
        Permite que o usuário baixe as previsões em um arquivo CSV.
        """
        if self.previsoes_futuras is None:
            messagebox.showwarning("Aviso", "Não há previsões para baixar.")
            return

        caminho = filedialog.asksaveasfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv")],
                                               title="Salvar Previsões")
        if caminho:
            try:
                df_previsoes = pd.DataFrame({
                    'Previsão': self.previsoes_futuras
                })
                df_previsoes.to_csv(caminho, index=False)
                messagebox.showinfo("Sucesso", f"Previsões salvas em {caminho}.")
                logger.info(f"Previsões salvas em {caminho}.")
            except Exception as e:
                logger.error(f"Erro ao salvar CSV: {str(e)}")
                messagebox.showerror("Erro", f"Erro ao salvar CSV: {str(e)}")

def main():
    """
    Função principal que inicia a aplicação Tkinter.
    """
    # Forçar TensorFlow a usar apenas a CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Garantir que a aplicação Tkinter seja executada na thread principal
    root = tk.Tk()
    app = AplicacaoTkinter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
