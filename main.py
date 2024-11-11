import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import ta
import datetime
import plotly.graph_objects as go
import pickle
from tensorflow.keras.models import load_model
import streamlit as st
from typing import Tuple


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

        # RobustScaler para lidar com outliers
        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

        self.historico_fitness = []
        self.melhor_modelo = None

    
    def criar_features_avancadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features técnicas avançadas com tratamento correto de dimensionalidade"""
        try:
            # Cópia do DataFrame para evitar modificações no original
            df = df.copy()
            
            # RSI
            rsi = ta.momentum.RSIIndicator(df['Close'])
            df['RSI'] = rsi.rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD_diff'] = macd.macd_diff()
            
            # ATR - Average True Range
            atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'])
            df['ATR'] = atr.average_true_range()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = df['BB_upper'] - df['BB_lower']
            
            # Médias Móveis
            for periodo in [5, 10, 20, 50]:
                df[f'SMA_{periodo}'] = ta.trend.sma_indicator(df['Close'], window=periodo)
                df[f'EMA_{periodo}'] = ta.trend.ema_indicator(df['Close'], window=periodo)
            
            # Volume
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Money Flow Index
            mfi = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'])
            df['MFI'] = mfi.money_flow_index()
            
            # Rate of Change
            roc = ta.momentum.ROCIndicator(df['Close'])
            df['ROC'] = roc.roc()
            
            # Log Returns e Volatilidade
            df['Log_return'] = np.log(df['Close']).diff()
            df['Volatility'] = df['Log_return'].rolling(window=20).std()
            
            # Remover linhas com valores NaN
            return df.dropna()
            
        except Exception as e:
            st.error(f"Erro ao criar indicadores técnicos: {str(e)}")
            raise

    def criar_sequencias_temporais(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(len(X) - self.janela_temporal):
            X_seq.append(X[i:(i + self.janela_temporal)])
            y_seq.append(y[i + self.janela_temporal])
        return np.array(X_seq), np.array(y_seq)

    def criar_modelo_hibrido(self):
        input_shape = (self.janela_temporal, X.shape[-1])

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

    def treinar_individuo(self, modelo, X_train, y_train, X_val, y_val):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]

        historia = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )

        return min(historia.history['val_loss'])

    def crossover_adaptativo(self, pai1, pai2):
        alpha = np.random.beta(0.5, 0.5)
        return [w1 * alpha + w2 * (1 - alpha) for w1, w2 in zip(pai1, pai2)]

    def mutacao_adaptativa(self, individuo, geracao):
        taxa = self.taxa_mutacao * (1 - geracao/self.num_geracoes)
        noise = np.random.normal(0, 0.1, size=individuo.shape)
        mask = np.random.random(individuo.shape) < taxa
        return np.where(mask, individuo + noise, individuo)

    def treinar(self, X, y):
        X_seq, y_seq = self.criar_sequencias_temporais(X, y)
        split = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        populacao = [self.criar_modelo_hibrido().get_weights() for _ in range(self.tamanho_populacao)]

        for geracao in range(self.num_geracoes):
            fitness_scores = []
            for pesos in populacao:
                modelo = self.criar_modelo_hibrido()
                modelo.set_weights(pesos)
                score = self.treinar_individuo(modelo, X_train, y_train, X_val, y_val)
                fitness_scores.append(1/(1 + score))

            indices_elite = np.argsort(fitness_scores)[-self.tamanho_elite:]
            nova_populacao = [populacao[i] for i in indices_elite]

            while len(nova_populacao) < self.tamanho_populacao:
                if np.random.random() < self.taxa_crossover:
                    idx1, idx2 = np.random.choice(len(populacao), 2, p=np.array(fitness_scores)/sum(fitness_scores))
                    filho = self.crossover_adaptativo(populacao[idx1], populacao[idx2])
                    filho = self.mutacao_adaptativa(filho, geracao)
                    nova_populacao.append(filho)

            populacao = nova_populacao[:self.tamanho_populacao]

            melhor_idx = np.argmax(fitness_scores)
            if not self.melhor_modelo or fitness_scores[melhor_idx] > max(self.historico_fitness or [0]):
                self.melhor_modelo = self.criar_modelo_hibrido()
                self.melhor_modelo.set_weights(populacao[melhor_idx])

            self.historico_fitness.append(max(fitness_scores))

    def prever(self, X):
        X_seq = np.array([X[i:i+self.janela_temporal] for i in range(len(X)-self.janela_temporal)])
        previsoes_norm = self.melhor_modelo.predict(X_seq)
        return self.scaler_y.inverse_transform(previsoes_norm)

    def prever_n_dias(self, X, n_dias):
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
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        dir_acc = np.mean((np.diff(y_true) * np.diff(y_pred)) > 0) * 100

        return {
            'MAPE': mape,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Direcional_Accuracy': dir_acc
        }

    def plotar_resultados(self, y_true, y_pred, titulo="Resultados de Previsão"):
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
            y=y_pred + 2*std_erro,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty'
        ))
        fig.add_trace(go.Scatter(
            y=y_pred - 2*std_erro,
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
        if self.melhor_modelo is None:
            raise ValueError("Nenhum modelo treinado para salvar")

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
        with open(f"{caminho}_info.pkl", 'rb') as f:
            modelo_info = pickle.load(f)

        instancia = cls(**modelo_info['parametros'])
        instancia.historico_fitness = modelo_info['historico_fitness']
        instancia.scaler_x = modelo_info['scaler_x']
        instancia.scaler_y = modelo_info['scaler_y']
        instancia.melhor_modelo = load_model(f"{caminho}_modelo.h5")
        return instancia

    def gerar_relatorio(self, y_true, y_pred, periodo=""):
        metricas = self.avaliar_modelo(y_true, y_pred)
        relatorio = f"""
        RELATÓRIO DE PERFORMANCE {periodo}
        ===============================
        
        Métricas Principais:
        - MAPE: {metricas['MAPE']:.2f}%
        - RMSE: {metricas['RMSE']:.4f}
        - R²: {metricas['R2']:.4f}
        - Acurácia Direcional: {metricas['Direcional_Accuracy']:.2f}%
        
        Análise de Erro:
        - Erro Médio: {np.mean(y_true - y_pred):.4f}
        - Erro Máximo: {np.max(np.abs(y_true - y_pred)):.4f}
        - Desvio Padrão do Erro: {np.std(y_true - y_pred):.4f}
        
        Estatísticas de Previsão:
        - Média Real: {np.mean(y_true):.4f}
        - Média Prevista: {np.mean(y_pred):.4f}
        - Correlação: {np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1]:.4f}
        """
        return relatorio
    
    def preparar_dados(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara os dados para treinamento"""
        try:
            # Adicionar features técnicas
            df = self.criar_features_avancadas(df)
            
            # Separar features e target
            features = df.drop(columns=['Close']).columns
            X = df[features].values
            y = df['Close'].values
            
            # Normalizar dados
            X = self.scaler_x.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            
            return X, y
            
        except Exception as e:
            st.error(f"Erro na preparação dos dados: {str(e)}")
            raise

    @staticmethod
    def interface_streamlit():
        st.set_page_config(page_title="GANN Trading", layout="wide")
        st.title("🧬 Sistema Avançado de Previsão com GANN")
        
        with st.sidebar:
            st.header("Configurações")
            ticker = st.text_input("Código da Ação", "PETR4.SA")
            data_inicio = st.date_input("Data Inicial", datetime.date(2020, 1, 1))
            data_fim = st.date_input("Data Final", datetime.date.today())
            
            st.header("Parâmetros do Modelo")
            pop_size = st.slider("Tamanho da População", 50, 200, 100)
            num_gen = st.slider("Número de Gerações", 50, 300, 150)
            mut_rate = st.slider("Taxa de Mutação", 0.1, 0.5, 0.2)
            
            dias_previsao = st.number_input("Dias para Previsão", 5, 30, 5)
        
        if st.button("Iniciar Análise"):
            try:
                with st.spinner("Carregando dados..."):
                    df = yf.download(ticker, start=data_inicio, end=data_fim)
                    if df.empty:
                        st.error("Não foi possível obter dados para o ticker informado")
                        return
                    
                    modelo = GANNMelhorado(
                        tamanho_populacao=pop_size,
                        num_geracoes=num_gen,
                        taxa_mutacao=mut_rate
                    )
                    
                    # Preparar dados
                    X, y = modelo.preparar_dados(df)
                    
                    # Mostrar dimensões dos dados
                    st.info(f"Dimensões dos dados - X: {X.shape}, y: {y.shape}")
                
                # Treinar modelo
                with st.spinner("Treinando modelo..."):
                    progress_bar = st.progress(0)
                    status = st.empty()
                    modelo.treinar(X, y)
                
                # Fazer previsões
                previsoes = modelo.prever(X)
                previsoes_futuras = modelo.prever_n_dias(X, dias_previsao)
                
                # Mostrar resultados
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Métricas de Performance")
                    metricas = modelo.avaliar_modelo(y, previsoes)
                    for nome, valor in metricas.items():
                        st.metric(nome, f"{valor:.4f}")
                
                with col2:
                    st.subheader("Previsões Futuras")
                    datas_futuras = pd.date_range(data_fim, periods=dias_previsao)
                    df_previsoes = pd.DataFrame({
                        'Data': datas_futuras,
                        'Previsão': previsoes_futuras.flatten()
                    })
                    st.dataframe(df_previsoes)
                
                # Gráficos
                st.plotly_chart(modelo.plotar_resultados(y, previsoes, f"Previsões para {ticker}"))
                
                # Relatório detalhado
                with st.expander("Ver Relatório Detalhado"):
                    st.text(modelo.gerar_relatorio(y, previsoes))
                    
            except Exception as e:
                st.error(f"Erro durante a execução: {str(e)}")
                raise

if __name__ == "__main__":
    GANNMelhorado.interface_streamlit()
