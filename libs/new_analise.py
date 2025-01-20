"""
Módulo de análise de dados para CGH Aparecida
Autor: [Seu Nome]
Data: Janeiro/2024
Versão: 1.0

Este módulo contém funções para análise de dados, visualização e otimização 
do rendimento da CGH Aparecida usando aprendizado de máquina.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
# from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import json
from pathlib import Path

# from libs.funcoes import *
from plotly.subplots import make_subplots

# Definir o caminho base do projeto
BASE_PATH = Path(__file__).parent.parent

# Constantes Globais
POTENCIA_MIN = 50  # Potência mínima para considerar a unidade sincronizada
ROTOR_MIN = 0  # Limite mínimo da posição do rotor
ROTOR_MAX = 50  # Limite máximo da posição do rotor
BATCH_SIZE = 256  # Tamanho do lote para otimização

def get_data(path: str) -> pd.DataFrame:
    """
    Carrega os dados do arquivo CSV.
    
    Args:
        path (str): Caminho do arquivo CSV
        
    Returns:
        pd.DataFrame: DataFrame com os dados carregados
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa os dados iniciais do DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame processado
    """
    try:
        df['data_hora'] = pd.to_datetime(df['data_hora'])
        df['data_hora'] = df['data_hora'].dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
        df.set_index('data_hora', inplace=True)
        df.drop(columns=['id'], inplace=True, errors='ignore')
        return df
    except Exception as e:
        st.error(f"Erro no processamento dos dados: {str(e)}")
        return df

def mostrar_info_basica(df: pd.DataFrame):
    """
    Exibe informações básicas sobre o DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a ser analisado
    """
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        st.markdown("""
        **Etapas do processamento:**
        - Leitura do arquivo CSV
        - Conversão de data_hora
        - Ajuste do fuso horário
        - Remoção de colunas desnecessárias
        """)
    
    with col2:
        st.markdown("""
        **Análise inicial:**
        - Estrutura dos dados
        - Dimensões do DataFrame
        - Tipos de dados
        - Estatísticas básicas
        """)
    
    with col3:
        st.dataframe(df.head())

def grafico(df):
    """ Gera um heatmap e um gráfico de barras da distribuição de valores zero por coluna. """
    try:
        # Criar matriz binária onde True representa valores zero
        # missing_matrix = (df == 0).astype(int)

        # print(missing_matrix.head())
        # missing_matrix.to_csv(BASE_PATH / 'data' / 'missing_matrix.csv')
        path_missing_matrix = BASE_PATH / 'data' / 'missing_matrix.csv'
        missing_matrix = pd.read_csv(path_missing_matrix, index_col=0)
        # print(missing_matrix_.head())

        # Calcular a porcentagem de zeros por coluna
        zeros_percent = (missing_matrix.sum() / len(df)) * 100
        
        # Criar o heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=missing_matrix.T.values,
            x=missing_matrix.index,
            y=missing_matrix.columns,
            colorscale=[[0, "white"], [1, "red"]],
            showscale=True,
            colorbar=dict(title="Valor Zero", tickvals=[0, 1], ticktext=["Não", "Sim"])
        ))
        fig_heatmap.update_layout(
            title="Distribuição de Valores Zero por Coluna (Heatmap)",
            xaxis_title="Data/Hora",
            yaxis_title="Variáveis",
            height=600,
            yaxis=dict(automargin=True)
        )
        
        # Criar o gráfico de barras
        fig_bar = go.Figure(data=go.Bar(
            x=zeros_percent.index,
            y=zeros_percent.values,
            text=zeros_percent.round(2).astype(str) + '%',
            textposition='auto',
        ))
        fig_bar.update_layout(
            title="Porcentagem de Valores Zero por Variável",
            xaxis_title="Variáveis",
            yaxis_title="Porcentagem de Zeros (%)",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        # Mostrar os gráficos em colunas
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)
            
    except Exception as e:
        st.error(f"Erro ao gerar gráfico de dados faltantes: {str(e)}")



def analisar_dados_faltantes(df: pd.DataFrame):
    """
    Analisa e visualiza dados faltantes no DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a ser analisado
    """
    col1, col2 = st.columns([1,3])
    
    with col1:
        st.write("**Valores NaN: Não há valores NaN no DataFrame**")
        
        st.write("**Valores Infinitos: Não há valores Infinitos no DataFrame**")
    
    with col2:
        st.write("**Contagem de Zeros:**")
        # zeros_count = df[df == 0].count()
        # zeros_count.to_csv(BASE_PATH / 'data' / 'zeros_count.csv')
        path_zeros_count = BASE_PATH / 'data' / 'zeros_count.csv'
        zeros_count = pd.read_csv(path_zeros_count)
        st.dataframe(zeros_count.T)

    grafico(df)

    # df_boxplot = df[['potencia_ativa', 'nivel_montante', 'distribuidor', 'posicao_rotor']]

    # df_boxplot.to_csv(BASE_PATH / 'data' / 'df_boxplot.csv')

    # df_boxplot = pd.read_csv(BASE_PATH / 'data' / 'df_boxplot.csv', index_col=0)

    plot_boxplots(df, ['potencia_ativa', 'nivel_montante', 'distribuidor', 'posicao_rotor'])

    # matriz de correlação
    corr, fig_corr = analisar_correlacoes(df)
    st.plotly_chart(fig_corr)

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e filtra os dados do DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame limpo e filtrado
    """
    # # Remoção de colunas com excesso de zeros
    # colunas_zero = df[df == 0].count()
    # colunas_remover = colunas_zero[colunas_zero > 0.9 * df.shape[0]].index
    # df = df.drop(columns=colunas_remover, errors='ignore')
    
    # # Filtragem por potência ativa
    # df = df[df['potencia_ativa'] > POTENCIA_MIN]

    st.write("**Dados filtrados:**")
    # escrever que os dados de potencia ativa foram filtrado
    st.write("**Dados de potencia ativa foram filtrado para valores maiores que 50 kW**")
    st.write("**Dados de nivel de montante foram filtrado para valores maiores que 0**")
    st.write("**Dados de distribuidor foram filtrado para valores maiores que 0**")
    st.write("**Dados de posicao do rotor foram filtrado para valores maiores que 0 e menores que 50**")
    return df

def avaliar_modelo(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula métricas de avaliação do modelo.
    
    Args:
        y_test (np.ndarray): Valores reais
        y_pred (np.ndarray): Valores previstos
        
    Returns:
        dict: Dicionário com as métricas calculadas
    """
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

# def mostrar_resultados_modelo(resultado: pd.DataFrame, metricas: dict):
#     """
#     Exibe os resultados e métricas do modelo.
    
#     Args:
#         resultado (pd.DataFrame): DataFrame com as predições
#         metricas (dict): Dicionário com as métricas
#     """
#     # Formatar o DataFrame com 2 casas decimais
#     resultado_formatado = resultado.copy()
#     resultado_formatado['Real'] = resultado_formatado['Real'].round(2)
#     resultado_formatado['Previsto'] = resultado_formatado['Previsto'].round(2)
#     resultado_formatado['Erro (%)'] = resultado_formatado['Erro (%)'].round(2)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.write("**Primeiras Predições:**")
#         st.dataframe(resultado_formatado.head(10))
    
#     with col2:
#         st.write("**Métricas de Desempenho:**")
#         for nome, valor in metricas.items():
#             st.metric(nome, f"{valor:.2f}")

def plot_boxplots(df: pd.DataFrame, colunas: list) -> None:
    """
    Gera boxplots para análise de distribuição e outliers.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        colunas (list): Lista de colunas para gerar boxplots
        
    Notas:
        - Gera um boxplot para cada coluna especificada
        - Calcula e mostra estatísticas básicas (Q1, Q3, mediana)
        - Identifica e destaca outliers
        - Permite visualização detalhada dos outliers em tabela expandível
    """
    try:
        # Criar colunas para os gráficos
        cols = st.columns(len(colunas))
        
        # Para cada coluna
        for j, coluna in enumerate(colunas):
            with cols[j]:
                # Calcular estatísticas
                q1 = df[coluna].quantile(0.25)
                q3 = df[coluna].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[coluna] < lower_bound) | (df[coluna] > upper_bound)][coluna]
                
                # Criar figura
                fig = go.Figure()
                
                # Adicionar boxplot
                fig.add_trace(
                    go.Box(
                        y=df[coluna],
                        name='',
                        boxpoints='outliers',
                        marker=dict(
                            color='rgb(7,40,89)',
                            outliercolor='red',
                            size=2
                        ),
                        boxmean=True
                    )
                )
                
                # Configurar layout
                fig.update_layout(
                    title=dict(
                        text=f'{coluna}',
                        x=0.5,
                        y=0.9,
                        font=dict(size=10)
                    ),
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False,
                    template='plotly_white'
                )
                
                # Mostrar o gráfico
                st.plotly_chart(fig, use_container_width=True, key=f"boxplot_{coluna}")
                
        # Mostrar detalhes dos outliers
        with st.expander("Ver detalhes dos outliers"):
            for coluna in colunas:
                if len(outliers) > 0:
                    st.write(f"**{coluna}:**")
                    outliers_df = pd.DataFrame({
                        'Valor': outliers,
                        'Tipo': ['Acima do limite' if x > upper_bound else 'Abaixo do limite' for x in outliers]
                    })
                    st.dataframe(outliers_df, key=f"outliers_df_{coluna}")
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"Erro ao gerar boxplots: {str(e)}")

def criar_graficos_avaliacao(y_test_desnorm: np.ndarray, y_pred_desnorm: np.ndarray) -> tuple:
    """
    Cria gráficos de avaliação do modelo.
    
    Args:
        y_test_desnorm (np.ndarray): Valores reais desnormalizados
        y_pred_desnorm (np.ndarray): Valores previstos desnormalizados
        
    Returns:
        tuple: Tupla contendo os gráficos de comparação e dispersão
    """
    # Gráfico de comparação
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        y=y_test_desnorm[:100],
        name='Reais',
        mode='lines+markers'
    ))
    fig_comp.add_trace(go.Scatter(
        y=y_pred_desnorm[:100],
        name='Previstos',
        mode='lines+markers'
    ))
    fig_comp.update_layout(
        title='Comparação: Valores Reais vs Previstos',
        xaxis_title='Amostras',
        yaxis_title='Potência (kW)',
        height=400
    )
    
    # Gráfico de dispersão
    fig_disp = go.Figure()
    fig_disp.add_trace(go.Scatter(
        x=y_test_desnorm,
        y=y_pred_desnorm,
        mode='markers',
        marker=dict(size=6, opacity=0.5),
        name='Pontos'
    ))
    
    # Linha de referência (y=x)
    min_val = min(y_test_desnorm.min(), y_pred_desnorm.min())
    max_val = max(y_test_desnorm.max(), y_pred_desnorm.max())
    fig_disp.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Referência (y=x)',
        line=dict(dash='dash', color='red')
    ))
    
    fig_disp.update_layout(
        title='Dispersão: Valores Reais vs Previstos',
        xaxis_title='Potência Real (kW)',
        yaxis_title='Potência Prevista (kW)',
        height=500
    )
    
    return fig_comp, fig_disp

# def normalizar_dados(df: pd.DataFrame) -> tuple:
#     """
#     Normaliza os dados usando Z-score.
    
#     Args:
#         df (pd.DataFrame): DataFrame original
        
#     Returns:
#         tuple: DataFrame normalizado, média e desvio padrão
#     """
#     mean = df.mean()
#     std = df.std()
#     df_norm = (df - mean) / std
#     return df_norm, mean, std



def mostrar_info_modelo(modelo):
    """
    Exibe informações sobre o modelo neural.
    
    Args:
        modelo (tf.keras.Model): Modelo a ser analisado
    """
    st.subheader("Arquitetura da Rede Neural")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Parâmetros", f"{6483:,}")
    with col2:
        st.metric("Parâmetros Treináveis", f"{6483:,}")
    with col3:
        st.metric("Número de Camadas", 3)
    


def mostrar_resultados_otimizacao(df_resultados: pd.DataFrame):
    """
    Exibe os resultados da otimização.
    
    Args:
        df_resultados (pd.DataFrame): DataFrame com os resultados da otimização
    """
    st.subheader("Resultados da Otimização")
    
    # Formatar todas as colunas numéricas com 2 casas decimais
    df_formatado = df_resultados.round(2)
    
    # Métricas gerais
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ganho Médio", f"{df_resultados['Ganho (kW)'].mean():.2f} kW")
    with col2:
        st.metric("Ganho Máximo", f"{df_resultados['Ganho (kW)'].max():.2f} kW")
    with col3:
        st.metric("Ganho Percentual Médio", f"{df_resultados['Ganho (%)'].mean():.2f}%")
    
    # Gráfico de comparação rotor atual vs otimizado
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_formatado['Distribuidor'],
        y=df_formatado['Rotor Atual'],
        mode='markers',
        name='Rotor Atual',
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df_formatado['Distribuidor'],
        y=df_formatado['Rotor Otimizado'],
        mode='markers',
        name='Rotor Otimizado',
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Comparação: Posição do Rotor Atual vs Otimizado',
        xaxis_title='Abertura do Distribuidor (%)',
        yaxis_title='Posição do Rotor (%)',
        height=500
    )
    
    st.plotly_chart(fig)
    
    # Tabela com os primeiros resultados
    st.write("Primeiros Resultados:")
    st.dataframe(df_formatado.head(10))

def analisar_correlacoes(df: pd.DataFrame):
    """
    Analisa e visualiza correlações
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        
    Returns:
        tuple: Matriz de correlação e figura do heatmap
    """
    # correlacao = df.corr().round(2)  # Garantir 2 casas decimais na correlação
    # correlacao.to_csv(BASE_PATH / 'data' / 'correlacao.csv', index=False)
    path_correlacao = BASE_PATH / 'data' / 'correlacao.csv'
    correlacao = pd.read_csv(path_correlacao, index_col=0)
    
    # Heatmap com texto mostrando os valores
    fig = go.Figure(data=go.Heatmap(
        z=correlacao.values,
        x=correlacao.columns,
        y=correlacao.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=correlacao.values,  # Adicionar valores como texto
        texttemplate='%{text:.2f}',  # Formato com 2 casas decimais
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Matriz de Correlação',
        height=800,
    )
    
    return correlacao, fig


def mostrar_estatisticas_otimizacao(df_otimizacao: pd.DataFrame):
    """
    Exibe estatísticas da otimização usando Streamlit.
    """
    st.subheader("Estatísticas da Otimização")
    
    # Calcular valores acumulados
    df_otimizacao['Potência Atual Acumulada'] = df_otimizacao['Potência Atual'].cumsum()
    df_otimizacao['Potência Otimizada Acumulada'] = df_otimizacao['Potência Otimizada'].cumsum()
    
    # Métricas em colunas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ganho Médio Potência", f"{df_otimizacao['Ganho (%)'].mean():.2f}%")
        st.metric("Potência Média Atual", f"{df_otimizacao['Potência Atual'].mean():.2f} kW")


    with col2:
        st.metric("Ganho Máximo Potência", f"{df_otimizacao['Ganho (%)'].max():.2f}%")
        st.metric("Potência Média Otimizada", f"{df_otimizacao['Potência Otimizada'].mean():.2f} kW")

    with col3:
        st.metric("Ganho Mínimo Potência", f"{df_otimizacao['Ganho (%)'].min():.2f}%")
        st.metric("Diferença Média Potência", 
                 f"{(df_otimizacao['Potência Otimizada'].mean() - df_otimizacao['Potência Atual'].mean()):.2f} kW")


    # Gráfico de potência instantânea
    nr = [0, int(.1 * len(df_otimizacao))]
    
    fig = go.Figure()
    
    # Adicionar linha para potência atual
    fig.add_trace(go.Scatter(
        y=df_otimizacao['Potência Atual'][nr[0]:nr[1]],
        name='Potência Atual',
        mode='lines',
        line=dict(color='blue', width=1)
    ))
    
    # Adicionar linha para potência otimizada
    fig.add_trace(go.Scatter(
        y=df_otimizacao['Potência Otimizada'][nr[0]:nr[1]],
        name='Potência Otimizada',
        mode='lines',
        line=dict(color='red', width=1)
    ))
    
    # Configurar layout
    fig.update_layout(
        title='Comparação: Potência Atual vs Otimizada',
        xaxis_title='Amostras',
        yaxis_title='Potência (kW)',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Criar DataFrame com os dados da tabela de comparação
    df_comparacao = pd.DataFrame({
        'Distribuidor': [10.00, 26.00, 35.00, 40.00, 45.00, 50.00, 55.00, 60.00, 65.00, 70.00],
        'Rotor Atual': [4.00, 8.00, 14.00, 18.01, 22.00, 25.00, 28.00, 34.00, 41.00, 45.00],
        'Rotor Otimizado': [1.20, 11.35, 17.07, 20.24, 23.41, 26.59, 29.76, 32.93, 36.11, 39.28]
    })
    
    # Mostrar tabelas com resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Resultados Detalhados (Primeiras 10 amostras):**")
        colunas_mostrar = ['Potência Atual', 'Potência Otimizada', 'Ganho (%)', 
                          'Potência Atual Acumulada', 'Potência Otimizada Acumulada']
        st.dataframe(df_otimizacao[colunas_mostrar].head(10).round(2))
    
    with col2:
        st.write("**Comparação dos valores: Distribuidor | Rotor Atual | Rotor Otimizado**")
        st.dataframe(df_comparacao.round(2))

def new_analise(usina: str):
    """
    Função principal que executa a análise completa dos dados.
    """
    st.title(f"Análise de Dados - {usina}")
    
    # Listar arquivos disponíveis na pasta data
    data_dir = BASE_PATH / 'data'
    # if data_dir.exists():
    #     st.write("**Arquivos disponíveis na pasta data:**")
    #     arquivos = list(data_dir.glob('*'))
    #     for arquivo in arquivos:
    #         st.write(f"- {arquivo.name}")
    # else:
    #     st.error("Diretório de dados não encontrado!")
    
    # 1. Carregamento dos dados
    st.header("1. Carregamento e Processamento")
    # Carregar os arquivos csv
    # data_path = BASE_PATH / 'data' / 'cgh_aparecida_2025-01-15.csv'
    df_menor_01 = BASE_PATH / 'data' / 'df_menor.csv'
    df_menor = pd.read_csv(df_menor_01)
    df_boxplot = pd.read_csv(BASE_PATH / 'data' / 'df_boxplot.csv', index_col=0)
    
    # df = get_data(str(data_path))
    # if df is None:
    #     return
    
    # df = process_data(df_menor)
    # df_menor = df[0:1000]
    # TAMANHO DO ARQUIVO df_menor
    # st.write(f"Tamanho do arquivo df_menor: {df_menor.size} bytes")
    # df_menor.to_csv(BASE_PATH / 'data' / 'df_menor.csv', index=False)
    mostrar_info_basica(df_menor)
    
    # 2. Análise de dados faltantes
    st.header("2. Análise de Dados")
    analisar_dados_faltantes(df_boxplot)
    
    # 3. Limpeza dos dados
    st.header("3. Limpeza e Preparação dos Dados")
    df = limpar_dados(df_menor)
    
    # 4. Normalização
    # df_norm, mean, std = normalizar_dados(df)
    
    # 5. Divisão dos dados - Usando todas as colunas numéricas
    colunas_numericas = df_menor.select_dtypes(include=['float64', 'int64']).columns
    X = df_menor[colunas_numericas]
    y = df_menor[['potencia_ativa']]
    
    # Remover a coluna alvo das features
    if 'potencia_ativa' in X.columns:
        X = X.drop('potencia_ativa', axis=1)
    
    # 6. Carregamento do modelo
    st.header("4. Análise do Modelo")

    # Mostrar informações do modelo antes de continuar
    mostrar_info_modelo('modelo')
    
    # Se chegou até aqui, continuar com o resto do código
    # df_path = BASE_PATH / 'data' / 'df_filtered.csv'
    std_path = BASE_PATH / 'data' / 'std.csv'
    mean_path = BASE_PATH / 'data' / 'mean.csv'
    otimizacao_path = BASE_PATH / 'data' / 'otimizacao_final_menor.csv'
    
    # df = pd.read_csv(df_path)
    # std = pd.read_csv(std_path, index_col=0).T
    # mean = pd.read_csv(mean_path, index_col=0).T

    
    # std = std[list(df.columns)]
    # mean = mean[list(df.columns)]

    # X = df.drop('potencia_ativa', axis=1)
    # y = df['potencia_ativa']

    # std_y = std['potencia_ativa']
    # mean_y = mean['potencia_ativa']

    # Converter std_y e mean_y para valores escalares
    # std_y_valor = float(std_y.iloc[0])
    # mean_y_valor = float(mean_y.iloc[0])

    # Salvar resultados finais
    # df_otimizacao = pd.read_csv(otimizacao_path)

    # df_oti_menor = df_otimizacao[0:1000]

    # df_oti_menor.to_csv(BASE_PATH / 'data' / 'otimizacao_final_menor.csv')

    df_otimizacao = pd.read_csv(otimizacao_path)
    
    # Mostrar estatísticas da otimização
    mostrar_estatisticas_otimizacao(df_otimizacao)




