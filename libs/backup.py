"""
CGH Aparecida - Módulo de Análise de Dados
----------------------------------------

Este módulo contém as funções necessárias para realizar análise de dados
e otimização da posição do rotor da CGH Aparecida.

Principais funcionalidades:
1. Carregamento e pré-processamento de dados
2. Análise exploratória
3. Tratamento de outliers
4. Análise de correlação
5. Modelagem preditiva
6. Otimização da posição do rotor

Autor: [Seu Nome]
Data: [Data Atual]
Versão: 1.0
"""

import pandas as pd
import streamlit as st
import numpy as np
from pytz import timezone
import missingno as msno
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import tensorflow as tf

# Constantes globais
POTENCIA_MINIMA = 50  # Potência mínima para considerar a unidade sincronizada
ROTOR_MIN = 0  # Limite mínimo da posição do rotor (%)
ROTOR_MAX = 50  # Limite máximo da posição do rotor (%)
CORRELACAO_FRACA = 0.3  # Limite para correlação fraca
CORRELACAO_FORTE = 0.7  # Limite para correlação forte


def carregar_modelo(path):
    ''' Carrega um modelo treinado de um arquivo '''

    try:
        # Carrega o modelo do arquivo
        model = tf.keras.models.load_model(path)

        return model

    except Exception as e:
        raise Exception(f'Erro ao carregar o modelo: {e}')

def subheader_abnt(text: str) -> None:
    """
    Cria um subtítulo formatado no padrão ABNT.
    
    Args:
        text (str): Texto do subtítulo
    """
    st.markdown(f'<p class="subheader-abnt">{text}</p>', unsafe_allow_html=True)

def interpolate_data() -> pd.DataFrame:
    """
    Retorna a tabela de interpolação Distribuidor x Rotor.
    
    Returns:
        pd.DataFrame: DataFrame com as colunas 'distribuidor' e 'posicao_rotor'
    """
    dados_interpolacao = [
        [10.00, 4.00], [26.00, 8.00], [35.00, 14.00],
        [40.00, 18.01], [45.00, 22.00], [50.00, 25.00],
        [55.00, 28.00], [60.00, 34.00], [65.00, 41.00],
        [70.00, 45.00]
    ]
    return pd.DataFrame(dados_interpolacao, columns=['distribuidor', 'posicao_rotor'])

def retirar_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove outliers de uma coluna usando o método IQR.
    
    Args:
        df (pd.DataFrame): DataFrame para análise
        
    Notas:
        - Gera um heatmap mostrando a distribuição temporal dos zeros
        - Gera um gráfico de barras com a porcentagem de zeros por coluna
        - Usa plotly para visualizações interativas
    """
    try:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    except Exception as e:
        raise Exception(f"Erro ao retirar outliers na coluna '{column}': {e}")

def grafico(df):
    """ Gera um heatmap e um gráfico de barras da distribuição de valores zero por coluna. """
    try:
        # Criar matriz binária onde True representa valores zero
        missing_matrix = (df == 0).astype(int)
        
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

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa o DataFrame aplicando transformações necessárias.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame processado
        
    Transformações aplicadas:
        1. Conversão da coluna data_hora para datetime
        2. Definição do fuso horário UTC
        3. Conversão para fuso horário America/Sao_Paulo
        4. Ordenação por data_hora
        5. Definição de data_hora como índice
        6. Remoção de colunas desnecessárias
    """
    try:
        # Converter para datetime se ainda não for
        df['data_hora'] = pd.to_datetime(df['data_hora'])
        
        # Definindo o fuso horário do UTC
        df['data_hora'] = df['data_hora'].dt.tz_localize('UTC')
        
        # Convertendo para o fuso horário do Brasil
        df['data_hora'] = df['data_hora'].dt.tz_convert('America/Sao_Paulo')
        
        # Ordenar o dataframe pela data_hora
        df = df.sort_values(by='data_hora')
        
        # Definir o index como data_hora
        df.set_index('data_hora', inplace=True)
        
        # Excluir colunas desnecessárias
        colunas_remover = ['id']
        df.drop(columns=colunas_remover, inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
        return df

def get_data(path):
    """ Carrega os dados de um arquivo CSV. """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Erro ao ler arquivo CSV: {str(e)}")
        return pd.DataFrame()

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

def normalizar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza os dados usando padronização Z-score.
    
    Args:
        df (pd.DataFrame): DataFrame com dados originais
        
    Returns:
        pd.DataFrame: DataFrame com dados normalizados
        
    Notas:
        - Usa normalização Z-score: (x - média) / desvio padrão
        - Resulta em dados com média 0 e desvio padrão 1
    """
    try:
        std = df.std()
        mean = df.mean()
        std.to_frame().to_csv('data/std.csv')
        mean.to_frame().to_csv('data/mean.csv')
        df_normalized = (df - mean) / std
        return df_normalized
    except Exception as e:
        st.error(f"Erro ao normalizar dados: {str(e)}")
        return df

def otimizar_rotor(df, modelo, x_test, y_test):
    """
    Otimiza a posição do rotor usando scipy.optimize.minimize
    """
    # Criar barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Desnormalizar valores para visualização
    distribuidor_mean = df['distribuidor'].mean()
    distribuidor_std = df['distribuidor'].std()
    rotor_mean = df['posicao_rotor'].mean()
    rotor_std = df['posicao_rotor'].std()
    potencia_mean = df['potencia_ativa'].mean()
    potencia_std = df['potencia_ativa'].std()

    # Lista para armazenar resultados
    resultados_otimizacao = []

    def objetivo(rotor_norm, x_atual):
        """Função objetivo para minimização (negativo da potência para maximizar)"""
        x_atual_copy = x_atual.copy()
        x_atual_copy.loc[:, 'posicao_rotor'] = rotor_norm
        return -modelo.predict(x_atual_copy)[0]

    # Total de pontos a processar
    total_pontos = len(x_test)
    
    # Para cada ponto no conjunto de teste
    for idx in range(total_pontos):
        try:
            # Atualizar barra de progresso
            progress = (idx + 1) / total_pontos
            progress_bar.progress(progress)
            status_text.text(f'Processando ponto {idx + 1} de {total_pontos} ({progress:.1%} concluído)')
            
            x_atual = x_test.iloc[[idx]]  # Manter como DataFrame com os nomes das colunas
            rotor_atual_norm = x_atual['posicao_rotor'].values[0]
            
            # Definir limites para o rotor (0 a 50% desnormalizado)
            bounds = [(0 - rotor_mean)/rotor_std, (50 - rotor_mean)/rotor_std]
            
            # Otimização usando diferentes pontos iniciais
            melhor_resultado = None
            melhor_potencia = float('-inf')
            
            # Testar diferentes pontos iniciais para evitar mínimos locais
            for rotor_inicial in np.linspace(bounds[0], bounds[1], 10):
                resultado = minimize(
                    objetivo,
                    x0=rotor_inicial,
                    args=(x_atual,),
                    method='L-BFGS-B',
                    bounds=[bounds],
                    options={'maxiter': 100}
                )
                
                potencia_pred = -resultado.fun
                if potencia_pred > melhor_potencia:
                    melhor_potencia = potencia_pred
                    melhor_resultado = resultado
            
            # Desnormalizar valores para registro
            distribuidor_real = x_atual['distribuidor'].values[0] * distribuidor_std + distribuidor_mean
            rotor_atual_real = rotor_atual_norm * rotor_std + rotor_mean
            rotor_otimo_real = melhor_resultado.x[0] * rotor_std + rotor_mean
            potencia_atual_real = y_test.iloc[idx] * potencia_std + potencia_mean
            potencia_otima_real = -melhor_resultado.fun * potencia_std + potencia_mean
            
            # Mostrar informações do ponto atual em uma única linha
            status_text.text(
                f'Ponto {idx + 1}/{total_pontos} - '
                f'Dist: {distribuidor_real:.1f}% | '
                f'Rotor: {rotor_atual_real:.1f}% → {rotor_otimo_real:.1f}% | '
                f'Ganho: {((potencia_otima_real - potencia_atual_real) / potencia_atual_real * 100):.1f}%'
            )

            resultados_otimizacao.append({
                'Distribuidor': round(distribuidor_real, 2),
                'Rotor Atual': round(rotor_atual_real, 2),
                'Rotor Ótimo': round(rotor_otimo_real, 2),
                'Potência Atual': round(potencia_atual_real, 2),
                'Potência Otimizada': round(potencia_otima_real, 2),
                'Ganho (%)': round(((potencia_otima_real - potencia_atual_real) / potencia_atual_real) * 100, 2),
                'Convergiu': melhor_resultado.success
            })
            
        except Exception as e:
            st.error(f"Erro no ponto {idx + 1}: {str(e)}")
            continue

    # Limpar a barra de progresso e o texto de status
    progress_bar.empty()
    status_text.empty()
    
    # Mostrar mensagem de conclusão
    st.success('Otimização concluída com sucesso!')
    
    return pd.DataFrame(resultados_otimizacao)

def new_analise(usina):
    st.title(f"Análise de Dados para CGH Aparecida")

    st.markdown("---")
    # Objetivo:
    st.markdown("""
    Análise de dados para a usina hidrelétrica CGH Aparecida, com o objetivo de otimizar o rendimento da usina. Usando
    a tabela de interpolação, Distribuidor X Rotor, que é a tabela de referência usada atualmente para a CGH Aparecida.
    Será realizado o treinamento de diferentes modelos de regressão para prever a potência ativa com base nas variáveis restantes.
    """)
    st.markdown("---")

    # 1. CARREGAMENTO DOS DADOS
    st.subheader("1. Carregamento e Visualização dos Dados")

    
    path = f'data/cgh_aparecida_2025-01-15.csv'
    df = get_data(path)
    df = process_data(df)

    subheader_abnt("1.1 - Visualização Inicial dos Dados")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:  
        st.markdown("""
        Nesta etapa, os dados do arquivo CSV são carregados e preparados para análise. As seguintes ações são realizadas:
        - Leitura do arquivo CSV contendo os dados da usina.
        - Conversão da coluna `data_hora` para o formato datetime.
        - Ajuste do fuso horário para 'America/Sao_Paulo'.
    - Definição da coluna `data_hora` como índice do DataFrame.
    - Remoção da coluna `id`, que não contribui para a análise.
    """)
    
    # st.markdown("---")
    with col2:  
        st.markdown("""
        Uma primeira olhada nos dados para entender sua estrutura e conteúdo:
        - Exibição de um exemplo dos dados carregados.
        - Apresentação das dimensões do DataFrame (número de linhas e colunas).
        - Detalhamento dos tipos de dados de cada coluna.
    - Estatísticas descritivas básicas para cada variável numérica.
    """)
    with col3:  
        subheader_abnt("1.2 - Amostra dos Dados")
        st.dataframe(df.head())


    subheader_abnt("1.3 - Tipos de Dados")
    tipos = df.dtypes.to_frame(name='Tipo').T
    st.dataframe(tipos)
    col1, col2 = st.columns([1,3])
    with col1:
        subheader_abnt("1.4 - Informações Gerais")
        st.write(f"Quantidade de linhas: {df.shape[0]}")
        st.write(f"Quantidade de colunas: {df.shape[1]}")
        st.write(f"Data inicial: {df.index[0]}")
        st.write(f"Data final: {df.index[-1]}")

    with col2:
        subheader_abnt("1.5 - Descrição dos Dados")
        descricao = df.describe()
        st.dataframe(descricao)


    # 2. ANÁLISE DE DADOS FALTANTES
    st.header("2. Análise de Dados Faltantes")

    st.markdown("""
    Investigação da presença de valores faltantes (zeros neste caso específico) no dataset:
    - Identificação de colunas com valores nulos (NaN).
    - Verificação de colunas com valores infinitos (Inf).
    - Cálculo da quantidade de valores iguais a 0 por coluna.
    - Visualização da distribuição de valores zero através de heatmap e gráfico de barras.
    """)
    subheader_abnt("2.1 - Detecção de Valores Faltantes")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Colunas com valores NaN:**")
        colunas_nan = df.columns[df.isna().any()].tolist()
        st.write(colunas_nan if colunas_nan else "Nenhuma coluna com NaN.")
        st.markdown("---")

        st.write("**Colunas com valores Inf:**")
        colunas_inf = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
        st.write(colunas_inf if colunas_inf else "Nenhuma coluna com Inf.")
        st.markdown("---")
    with col2:
        st.write("**Quantidade de valores igual a 0 por coluna:**")
        colunas_zero = df[df == 0].count()
        st.dataframe(colunas_zero.to_frame(name='Contagem').T)
        st.markdown("---")

    st.subheader("3.2 - Visualização de Dados Faltantes")
    grafico(df)
    st.markdown("---")

    # 4. LIMPEZA DOS DADOS
    st.header("4. Limpeza e Filtragem dos Dados")
    st.markdown("---")
    st.markdown("""
    Etapa de limpeza para refinar o dataset para uma análise mais precisa:
    - Remoção de colunas onde a maioria dos valores são zero (acima de 90%).
    - Filtragem das linhas para considerar apenas quando a unidade geradora está sincronizada (potência ativa > 50).
    """)
    subheader_abnt("4.1 - Remoção de Colunas com Excesso de Zeros")
    colunas_zero_percent = df[df == 0].count()
    colunas_para_remover = colunas_zero_percent[colunas_zero_percent > 0.9 * df.shape[0]].index.tolist()
    st.write("**Colunas removidas devido ao excesso de zeros:**")
    st.write(colunas_para_remover if colunas_para_remover else "Nenhuma coluna removida por excesso de zeros.")
    df_copy = df.copy()
    df.drop(columns=colunas_para_remover, inplace=True, errors='ignore')
    st.markdown("---")

    st.subheader("4.2 - Filtragem por Potência Ativa")
    st.write("**Registros antes da filtragem:**", df_copy.shape[0])
    df = df[df['potencia_ativa'] > 50]
    st.write("**Registros após a filtragem (potência ativa > 50):**", df.shape[0])
    st.markdown("---")

    # 5. ANÁLISE DE OUTLIERS
    st.header("5. Análise e Remoção de Outliers")
    st.markdown("---")
    st.markdown("""
    Identificação e tratamento de valores atípicos nas principais variáveis:
    - Visualização da distribuição e detecção de outliers usando boxplots.
    - Remoção de outliers da coluna de potência ativa para evitar distorções na análise.
    """)
    subheader_abnt("5.1 - Boxplots para Detecção de Outliers")
    plot_boxplots(df, ['potencia_ativa', 'nivel_montante', 'distribuidor', 'posicao_rotor'])
    st.markdown("---")

    st.subheader("5.2 - Remoção de Outliers da Potência Ativa")
    df_copy_outliers = df.copy()
    df = retirar_outliers(df, 'potencia_ativa')
    st.write(f"Registros removidos por serem considerados outliers: {df_copy_outliers.shape[0] - df.shape[0]}")
    st.markdown("---")

    # 6. NORMALIZAÇÃO DOS DADOS
    st.header("6. Normalização dos Dados")
    st.markdown("---")
    st.markdown("""
    Normalização das variáveis para que tenham média zero e desvio padrão um, facilitando a comparação e análises subsequentes.
    """)
    st.subheader("Dados normalizados:")
    df_normalized = normalizar_dados(df)
    st.dataframe(df_normalized.head())
    st.markdown("---")

    # 7. ANÁLISE DE CORRELAÇÃO
    st.header("7. Análise de Correlação")
    st.markdown("---")
    st.markdown("""
    Investigação das relações lineares entre as variáveis através da matriz de correlação e sua visualização em um heatmap.
    """)
    subheader_abnt("7.1 - Matriz de Correlação")
    correlacao, fig_corr = analisar_correlacoes(df_normalized)
    st.dataframe(correlacao)
    st.plotly_chart(fig_corr)
    st.markdown("---")

    # 8. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)
    # aplicar_pca(df_normalized)

    # 9. FILTRAGEM POR CORRELAÇÃO
    st.header("9. Filtragem por Correlação")
    st.markdown("---")
    st.markdown("""
    Análise e filtragem de variáveis baseada em seus coeficientes de correlação com a potência ativa:
    - Correlação fraca: |r| < 0.3
    - Correlação moderada: 0.3 ≤ |r| < 0.7
    - Correlação forte: |r| ≥ 0.7
    """)

    # Criar máscara para diferentes níveis de correlação
    correlacao_abs = correlacao.abs()
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.write("**9.1 - Pares com correlação forte (|r| ≥ 0.7):**")
        pares_fortes = []
        for i in range(len(correlacao.columns)):
            for j in range(i+1, len(correlacao.columns)):
                if abs(correlacao.iloc[i,j]) >= 0.7:
                    pares_fortes.append({
                        'Variável 1': correlacao.columns[i],
                        'Variável 2': correlacao.columns[j],
                        'Correlação': correlacao.iloc[i,j]
                    })
        if pares_fortes:
            st.dataframe(pd.DataFrame(pares_fortes))
        else:
            st.write("Nenhum par encontrado")
    
    with col2:
        st.write("**9.2 - Variáveis com correlações fracas em relação à potência ativa (|r| < 0.3):**")
        # Pegar apenas as correlações com potência ativa
        correlacoes_potencia = correlacao_abs['potencia_ativa']
        # Filtrar correlações fracas
        vars_fracas = correlacoes_potencia[correlacoes_potencia < 0.3]
        if not vars_fracas.empty:
            # Criar DataFrame com as correlações fracas
            df_fracas = pd.DataFrame({
                'Variável': vars_fracas.index,
                'Correlação': vars_fracas.values
            }).round(3)
            st.dataframe(df_fracas)
        else:
            st.write("Nenhuma variável com correlação fraca encontrada")

    # Filtrar o DataFrame mantendo apenas variáveis com correlação significativa com potência ativa
    vars_para_manter = correlacoes_potencia[correlacoes_potencia >= 0.3].index
    df_filtered = df_normalized[vars_para_manter]

    df_filtered.to_csv('data/df_filtered.csv', index=False)
    
    st.write("**9.3 - DataFrame após remoção de variáveis com correlações fracas com potência ativa:**")
    st.dataframe(df_filtered)
    
    # Mostrar dimensões antes e depois
    st.write(f"Dimensões originais: {df_copy.shape}")
    st.write(f"Dimensões após filtragem: {df_filtered.shape}")

    # 10. TREINAMENTO DO MODELO
    st.header("10. Treinamento do Modelo")
    st.markdown("---")
    st.markdown("""
    Treinamento do modelo de regressão linear para prever a potência ativa com base nas variáveis restantes.
    O objetivo é criar um modelo que possa prever a potência ativa com base nas variáveis restantes. Depois,
    alterar os valores da variável rotor para maximizar a potência ativa. A tabela abaixo é a tabela de interpolação,
    Distribuidor X Rotor, que é a tabela de referência usada atualmente para a CGH Aparecida.
    """)

    tabela_rotor = interpolate_data()
    st.markdown("Tabela de interpolação, Distribuidor X Rotor:")
    st.dataframe(tabela_rotor)
    st.markdown("---")

    # definir a variável x e y para o treinamento do modelo
    x = df_filtered.drop(columns=['potencia_ativa'])
    y = df_filtered['potencia_ativa']

    # Dividir os dados em conjuntos de treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # carregar rede neural treinada
    path = 'C:/projetos/engesep/5_dist_rotor/data/data/modelo_rede_neural.keras'
    modelo_rede_neural = carregar_modelo(path)

    # 11. APLICAÇÃO DO MODELO
    st.header("11. Aplicação do Modelo")
    st.markdown("---")
    st.markdown("""
    Aplicação da rede neural para prever a potência ativa com base nas variáveis restantes.
    Análise das predições e métricas de desempenho do modelo.
    """)

    col1, col2 = st.columns(2)
    with col1:
        # Aplicar o modelo para prever a potência ativa
        y_pred = modelo_rede_neural.predict(x_test)
        
        # Usar os valores de std e mean do DataFrame original
        std_potencia = df['potencia_ativa'].std()
        mean_potencia = df['potencia_ativa'].mean()
        
        # Desnormalizar os valores
        y_pred_desnorm = y_pred * std_potencia + mean_potencia
        y_test_desnorm = np.array(y_test) * std_potencia + mean_potencia
        
        resultado = pd.DataFrame({
            'Potência Real': y_test_desnorm,
            'Potência Prevista': y_pred_desnorm.flatten(),
            'Diferença': y_pred_desnorm.flatten() - y_test_desnorm,
            'Erro (%)': ((y_pred_desnorm.flatten() - y_test_desnorm) / y_test_desnorm) * 100
        })
        
        st.write("**Primeiras predições do modelo:**")
        st.dataframe(resultado.head(10))
        
    with col2:
        # Calcular métricas
        mae = np.mean(np.abs(resultado['Diferença']))
        rmse = np.sqrt(np.mean(resultado['Diferença']**2))
        erro_medio_percentual = np.mean(np.abs(resultado['Erro (%)']))
        
        st.write("**Métricas de Desempenho:**")
        st.metric("Erro Médio Absoluto (MAE)", f"{mae:.2f} kW")
        st.metric("Erro Quadrático Médio (RMSE)", f"{rmse:.2f} kW")
        st.metric("Erro Médio Percentual", f"{erro_medio_percentual:.2f}%")

    # Visualizações
    st.subheader("11.1 - Visualização das Predições")
    
    # Gráfico de comparação
    fig_comp, fig_disp = criar_graficos_avaliacao(
        resultado, 
        resultado['Potência Real'], 
        resultado['Potência Prevista']
    )
    st.plotly_chart(fig_comp)
    st.plotly_chart(fig_disp)

    # Histograma dos erros
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=resultado['Erro (%)'],
        nbinsx=50,
        name='Erro'
    ))
    fig_hist.update_layout(
        title='Distribuição dos Erros Percentuais',
        xaxis_title='Erro (%)',
        yaxis_title='Frequência',
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)

def analisar_correlacoes(df: pd.DataFrame):
    """Analisa e visualiza correlações"""
    correlacao = df.corr().round(2)
    
    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlacao.values,
        x=correlacao.columns,
        y=correlacao.index,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig.update_layout(title='Matriz de Correlação', height=600)
    return correlacao, fig

def criar_graficos_avaliacao(resultado, y_test_desnorm, y_pred_desnorm):
    """Cria gráficos de avaliação do modelo"""
    # Gráfico de comparação
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        y=y_test_desnorm[:100], 
        name='Reais',
        mode='lines+markers'
    ))
    fig_comp.add_trace(go.Scatter(
        y=y_pred_desnorm[:100], # Removido o flatten()
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
        y=y_pred_desnorm, # Removido o flatten()
        mode='markers',
        marker=dict(
            size=6,
            opacity=0.5
        ),
        name='Pontos'
    ))
    
    # Adicionar linha de referência (y=x)
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

if __name__ == "__main__":
    new_analise('CGH Aparecida')





