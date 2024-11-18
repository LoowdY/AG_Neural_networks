import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, GRU, Input
from tensorflow.keras.optimizers import Adam
import ta
import datetime
import plotly.graph_objects as go
import pickle
import streamlit as st
from typing import Tuple, Callable, Optional
from multiprocessing import Pool, cpu_count
import warnings
# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')


def criar_modelo_hibrido(input_shape):
    modelo = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        GRU(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    modelo.compile(optimizer=optimizer, loss='huber')
    return modelo


def treinar_individuo(pesos, input_shape, X_val, y_val):
    modelo = criar_modelo_hibrido(input_shape)
    modelo.set_weights(pesos)
    try:
        loss = modelo.evaluate(X_val, y_val, verbose=0)
        return loss
    except Exception:
        return float('inf')


@st.cache_resource
def get_pool():
    return Pool(processes=cpu_count())


class GANNMelhorado:
    def __init__(
            self,
            tamanho_populacao: int = 100,
            num_geracoes: int = 150,
            taxa_mutacao: float = 0.2,
            taxa_crossover: float = 0.85,
            tamanho_elite: int = 4,
            janela_temporal: int = 10
    ):
        self.tamanho_populacao = tamanho_populacao
        self.num_geracoes = num_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.tamanho_elite = tamanho_elite
        self.janela_temporal = janela_temporal

        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

        self.historico_fitness = []
        self.melhor_modelo = None

    def crossover_adaptativo(self, pai1, pai2):
        """Realiza crossover adaptativo entre dois conjuntos de pesos"""
        filho = []
        for p1, p2 in zip(pai1, pai2):
            if np.random.random() < self.taxa_crossover:
                # Crossover aritmético adaptativo
                alpha = np.random.random()  # Taxa de mistura
                peso_filho = alpha * p1 + (1 - alpha) * p2
            else:
                # Seleciona um dos pais aleatoriamente
                peso_filho = p1 if np.random.random() < 0.5 else p2
            filho.append(peso_filho)
        return filho

    def mutacao_adaptativa(self, individuo, geracao):
        """Realiza mutação adaptativa nos pesos"""
        # Taxa de mutação adaptativa que diminui com as gerações
        taxa_atual = self.taxa_mutacao * (1 - geracao / self.num_geracoes)

        for i in range(len(individuo)):
            if np.random.random() < taxa_atual:
                # Magnitude da mutação diminui com as gerações
                magnitude = np.random.normal(0, 0.1 * (1 - geracao / self.num_geracoes))
                individuo[i] = individuo[i] + magnitude * np.random.randn(*individuo[i].shape)
        return individuo

    def criar_features_avancadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates advanced technical features ensuring inputs are one-dimensional Series"""
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

            # Moving Averages
            for periodo in [5, 10, 20, 50]:
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

            # Log Returns and Volatility
            df['Log_return'] = np.log(close).diff()
            df['Volatility'] = df['Log_return'].rolling(window=20).std()

            return df.dropna()

        except Exception as e:
            st.error(f"Error creating technical indicators: {str(e)}")
            raise

    def criar_sequencias_temporais(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates time sequences for temporal analysis"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.janela_temporal):
            X_seq.append(X[i:(i + self.janela_temporal)])
            y_seq.append(y[i + self.janela_temporal])
        return np.array(X_seq), np.array(y_seq)

    def treinar(self, X, y, callback: Optional[Callable[[int, float], None]] = None):
        """Trains the model using genetic algorithm optimization"""
        X_seq, y_seq = self.criar_sequencias_temporais(X, y)
        split = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        input_shape = (self.janela_temporal, X.shape[-1])
        populacao = [criar_modelo_hibrido(input_shape).get_weights() for _ in range(self.tamanho_populacao)]

        pool = get_pool()

        for geracao in range(self.num_geracoes):
            fitness_scores = pool.starmap(
                treinar_individuo,
                [(pesos, input_shape, X_val, y_val) for pesos in populacao]
            )

            fitness_scores = [1 / (1 + loss) if loss != float('inf') else 0 for loss in fitness_scores]

            elite_indices = np.argsort(fitness_scores)[-self.tamanho_elite:]
            nova_populacao = [populacao[i] for i in elite_indices]

            while len(nova_populacao) < self.tamanho_populacao:
                if np.random.random() < self.taxa_crossover:
                    total_fitness = sum(fitness_scores)
                    probs = [f / total_fitness for f in fitness_scores]
                    idx1, idx2 = np.random.choice(len(populacao), 2, p=probs)
                    pai1, pai2 = populacao[idx1], populacao[idx2]
                    filho = self.crossover_adaptativo(pai1, pai2)
                    filho = self.mutacao_adaptativa(filho, geracao)
                    nova_populacao.append(filho)

            populacao = nova_populacao[:self.tamanho_populacao]

            melhor_idx = np.argmax(fitness_scores)
            melhor_fitness = fitness_scores[melhor_idx]
            self.historico_fitness.append(melhor_fitness)

            if not self.melhor_modelo or melhor_fitness > max(self.historico_fitness[:-1] or [0]):
                self.melhor_modelo = criar_modelo_hibrido(input_shape)
                self.melhor_modelo.set_weights(populacao[melhor_idx])

            if callback:
                callback(geracao + 1, melhor_fitness)

    def prever(self, X):
        """Makes predictions using the best model"""
        X_seq = np.array([X[i:i + self.janela_temporal] for i in range(len(X) - self.janela_temporal)])
        previsoes_norm = self.melhor_modelo.predict(X_seq)
        return self.scaler_y.inverse_transform(previsoes_norm)

    def prever_n_dias(self, X, n_dias):
        """Makes predictions for n days ahead"""
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
        """Evaluates model performance"""
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
        """Plots prediction results with confidence intervals"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=y_true,
            name='Real',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=y_pred,
            name='Previsto',
            line=dict(color='red', width=2)
        ))
        erro = y_true - y_pred
        std_erro = np.std(erro)
        fig.add_trace(go.Scatter(
            y=y_pred + 2 * std_erro,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty'
        ))
        fig.add_trace(go.Scatter(
            y=y_pred - 2 * std_erro,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty',
            name='Intervalo de Confiança'
        ))
        fig.update_layout(
            title=titulo,
            xaxis_title='Tempo',
            yaxis_title='Preço',
            showlegend=True,
            template='plotly_white'
        )
        return fig

    def salvar_modelo(self, caminho):
        """Saves the model and its parameters"""
        if self.melhor_modelo is None:
            raise ValueError("No trained model to save")

        modelo_info = {
            'parametros': {
                'tamanho_populacao': self.tamanho_populacao,
                'num_geracoes': self.num_geracoes,
                'taxa_mutacao': self.taxa_mutacao,
                'taxa_crossover': self.taxa_crossover,
                'tamanho_elite': self.tamanho_elite,
                'janela_temporal': self.janela_temporal
            },
            'historico_fitness': self.historico_fitness,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y
        }

        self.melhor_modelo.save(f"{caminho}_modelo.h5")
        with open(f"{caminho}_info.pkl", 'wb') as f:
            pickle.dump(modelo_info, f)

    @classmethod
    def carregar_modelo(cls, caminho):
        """Loads a saved model"""
        with open(f"{caminho}_info.pkl", 'rb') as f:
            modelo_info = pickle.load(f)

        instancia = cls(**modelo_info['parametros'])
        instancia.historico_fitness = modelo_info['historico_fitness']
        instancia.scaler_x = modelo_info['scaler_x']
        instancia.scaler_y = modelo_info['scaler_y']
        instancia.melhor_modelo = load_model(f"{caminho}_modelo.h5")
        return instancia

    def gerar_relatorio(self, y_true, y_pred, periodo=""):
            """Generates a detailed performance report"""
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

    def preparar_dados(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            """Prepara os dados para treinamento"""
            try:
                df = self.criar_features_avancadas(df)

                features = df.drop(columns=['Close']).columns
                X = df[features].values
                y = df['Close'].values

                X = self.scaler_x.fit_transform(X)
                y = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

                return X, y

            except Exception as e:
                st.error(f"Erro na preparação dos dados: {str(e)}")
                raise

    @staticmethod
    def interface_streamlit():
            """Interface Streamlit para o sistema GANN"""
            st.set_page_config(page_title="GANN Trading", layout="wide")
            st.title("🧬 Sistema Avançado de Previsão com GANN")

            with st.sidebar:
                st.header("Configurações")
                ticker = st.text_input("Código da Ação", "PETR4.SA")
                data_inicio = st.date_input("Data Inicial", datetime.date(2020, 1, 1))
                data_fim = st.date_input("Data Final", datetime.date.today())

                st.header("Parâmetros do Algoritmo Genético")
                pop_size = st.slider("Tamanho da População", 10, 200, 100)
                num_gen = st.slider("Número de Gerações", 10, 300, 150)
                mut_rate = st.slider("Taxa de Mutação", 0.01, 0.5, 0.2)
                crossover_rate = st.slider("Taxa de Crossover", 0.1, 1.0, 0.85)
                elite_size = st.slider("Tamanho da Elite", 1, 20, 4)
                janela_temporal = st.slider("Janela Temporal", 5, 30, 10)

                st.header("Parâmetros de Previsão")
                dias_previsao = st.selectbox("Dias para Previsão", options=[1, 2, 3, 4], index=0)

            if st.button("Iniciar Análise"):
                try:
                    with st.spinner("Carregando dados..."):
                        tickers = [t.strip() for t in ticker.split(',')]
                        if len(tickers) > 1:
                            st.warning("Apenas o primeiro ticker será processado.")
                        ticker_single = tickers[0]

                        df = yf.download(ticker_single, start=data_inicio, end=data_fim)
                        if df.empty:
                            st.error("Não foi possível obter dados para o ticker informado")
                            st.stop()

                        required_columns = ['Close', 'High', 'Low', 'Volume']
                        for col in required_columns:
                            if col not in df.columns:
                                st.error(f"Coluna '{col}' ausente nos dados baixados.")
                                st.stop()

                        modelo = GANNMelhorado(
                            tamanho_populacao=pop_size,
                            num_geracoes=num_gen,
                            taxa_mutacao=mut_rate,
                            taxa_crossover=crossover_rate,
                            tamanho_elite=elite_size,
                            janela_temporal=janela_temporal
                        )

                        X, y = modelo.preparar_dados(df)
                        st.info(f"Dimensões dos dados - X: {X.shape}, y: {y.shape}")

                    # Inicializar áreas para logs
                    log_placeholder = st.empty()
                    fitness_placeholder = st.empty()

                    def callback(geracao, melhor_fitness):
                        log_placeholder.text(
                            f"Geração {geracao}/{modelo.num_geracoes} - Melhor Fitness: {melhor_fitness:.6f}")
                        fitness_placeholder.line_chart(modelo.historico_fitness)

                    with st.spinner("Treinando modelo..."):
                        modelo.treinar(X, y, callback=callback)

                    with st.spinner("Realizando previsões..."):
                        previsoes = modelo.prever(X)
                        previsoes_futuras = modelo.prever_n_dias(X, dias_previsao)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Métricas de Performance")
                        y_true_evaluation = y[-len(previsoes):]
                        y_pred_evaluation = previsoes.flatten()
                        metricas = modelo.avaliar_modelo(y_true_evaluation, y_pred_evaluation)
                        st.metric("MAPE", f"{metricas['MAPE']:.2f}%")
                        st.metric("RMSE", f"{metricas['RMSE']:.4f}")
                        st.metric("MSE", f"{metricas['MSE']:.4f}")

                    with col2:
                        st.subheader("Previsões Futuras")
                        datas_futuras = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=dias_previsao)
                        df_previsoes = pd.DataFrame({
                            'Data': datas_futuras,
                            'Previsão': previsoes_futuras.flatten()
                        })
                        st.dataframe(df_previsoes)

                    st.plotly_chart(modelo.plotar_resultados(
                        y_true_evaluation,
                        y_pred_evaluation,
                        f"Previsões para {ticker_single}"
                    ))

                    with st.expander("Ver Relatório Detalhado"):
                        st.text(modelo.gerar_relatorio(y_true_evaluation, y_pred_evaluation, periodo=""))

                    # Opção para salvar o modelo
                    if st.button("Salvar Modelo"):
                        try:
                            modelo.salvar_modelo(f"modelo_{ticker_single}")
                            st.success("Modelo salvo com sucesso!")
                        except Exception as e:
                            st.error(f"Erro ao salvar o modelo: {str(e)}")

                except Exception as e:
                    st.error(f"Erro durante a execução: {str(e)}")
                    raise

if __name__ == "__main__":
    GANNMelhorado.interface_streamlit()
