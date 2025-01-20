'''
Rede Neural Artificial
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Configurações globais
LIMITE_AMOSTRAS = 10000  # Número máximo de amostras para otimização
BATCH_SIZE = 64  # Tamanho do lote para processamento
LIMITE_DADOS = 1
INTERACOES = 40
EPOCAS = 20


path = ""
# Carregar os dados
df = pd.read_csv(r'C:\projetos\engesep\5_dist_rotor\data\df_filtered.csv')
std = pd.read_csv(r'C:\projetos\engesep\5_dist_rotor\data\std.csv', index_col=0).T
mean = pd.read_csv(r'C:\projetos\engesep\5_dist_rotor\data\mean.csv', index_col=0).T


print(df.shape[0], int(df.shape[0] * LIMITE_DADOS))
df = df[0:int(df.shape[0] * LIMITE_DADOS)]

print(f'Quantidade de dados: {len(df)}')
# print(std.head())
# print(mean.head())

std = std[list(df.columns)]
mean = mean[list(df.columns)]

# print(df.head())
# print(df.info())
# print(df.describe())
# Separar as variáveis independentes (X) e a variável dependente (y)
X = df.drop('potencia_ativa', axis=1)
y = df['potencia_ativa']

std_y = std['potencia_ativa']
mean_y = mean['potencia_ativa']

# Converter std_y e mean_y para valores escalares
std_y_valor = float(std_y.iloc[0])
mean_y_valor = float(mean_y.iloc[0])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# LIMITE_AMOSTRAS = len(X_test)

# print(X_train.columns)
# print(y_train.to_frame().columns)


def modelo_rede_neural():
    with tf.device('/GPU:0'):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])
        
        return model

print(tf.config.list_physical_devices('GPU'))

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

# usar context com gpu
with tf.device('/GPU:0'):

    def train_model():

        global X_train, y_train, X_test, y_test, std_y_valor, mean_y_valor
        modelo = modelo_rede_neural()

        print(modelo.summary())

        modelo.fit(X_train, y_train, epochs=EPOCAS, verbose=1)


        # Salvar o modelo
        modelo.save(r'C:\projetos\engesep\5_dist_rotor\data\data\modelo_rede_neural.keras')

        # avaliar o modelo
        loss, mae = modelo.evaluate(X_test, y_test, verbose=1)
        print(f'Loss: {loss:.4f}, MAE: {mae:.4f}')

        # prever os valores
        y_pred = modelo.predict(X_test)

        # # Converter std_y e mean_y para valores escalares
        # std_y_valor = float(std_y.iloc[0])
        # mean_y_valor = float(mean_y.iloc[0])

        # desnormalizar os valores usando os valores escalares
        y_pred = y_pred * std_y_valor + mean_y_valor
        y_test = np.array(y_test) * std_y_valor + mean_y_valor

        # print(y_pred.shape)
        # print(y_test.shape)

        # Criar DataFrame para melhor visualização dos resultados
        resultados = pd.DataFrame({
            'Valor Real': y_test,
            'Valor Predito': y_pred.flatten(),
            'Diferença': y_test - y_pred.flatten()
        })

        print("\nPrimeiras 10 predições:")
        print(resultados.head(100))

        # Calcular métricas
        mae = np.mean(np.abs(resultados['Diferença']))
        mse = np.mean(resultados['Diferença']**2)
        rmse = np.sqrt(mse)

        print(f"\nMétricas de Erro:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = r'C:\projetos\engesep\5_dist_rotor\data\rede_neural_A.png'
        name_a = r'C:\projetos\engesep\5_dist_rotor\data\rede_neural_B.png'
        # plotar os valores
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Valor real', alpha=0.7)
        plt.plot(y_pred, label='Valor predito', alpha=0.7)
        plt.title('Comparação entre Valores Reais e Preditos')
        plt.xlabel('Amostras')
        plt.ylabel('Potência Ativa')
        plt.legend()
        plt.grid(True)
        plt.savefig(name)
        plt.close()

        # Plotar gráfico de dispersão
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Gráfico de Dispersão: Valores Reais vs Preditos')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Preditos')
        plt.grid(True)
        plt.savefig(name_a)
        plt.close()

        return modelo
    
    modelo = train_model()

    # Após calcular as métricas, adicionar análise de otimização:
    print("\nAnálise de Otimização do Rotor usando TensorFlow:")
    
    @tf.function
    def predict_potencia_tf(x_input):
        """Função de predição otimizada para TensorFlow"""
        return modelo(x_input)

    def otimizar_rotor_tf(x_sample):
        """Otimiza a posição do rotor usando otimizador do TensorFlow"""
        # Converter amostra para tensor
        x_tensor = tf.convert_to_tensor(x_sample.to_numpy(), dtype=tf.float32)
        
         # Criar variável para posição do rotor com limite inferior em 0
        rotor_inicial = x_tensor[0, X_test.columns.get_loc('posicao_rotor')]
        rotor_inicial = tf.maximum(rotor_inicial, 0.0)  # Garante valor inicial não negativo
        rotor_var = tf.Variable(rotor_inicial, dtype=tf.float32)
        
        # Otimizador com learning rate maior para convergência mais rápida
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
        
        # Histórico de valores
        best_potencia = float('-inf')
        best_rotor = None
        
        # Reduzir número de iterações de 100 para 50
        for _ in range(INTERACOES):
            with tf.GradientTape() as tape:
                x_mod = tf.identity(x_tensor)
                x_mod = tf.tensor_scatter_nd_update(
                    x_mod, 
                    [[0, X_test.columns.get_loc('posicao_rotor')]], 
                    [rotor_var]
                )
                
                potencia_norm = predict_potencia_tf(x_mod)[0, 0]
                potencia = potencia_norm * std_y_valor + mean_y_valor
                
                rotor_desnorm = rotor_var * std['posicao_rotor'].iloc[0] + mean['posicao_rotor'].iloc[0]
                penalty = 1000.0 * tf.maximum(0.0, -rotor_desnorm) + 1000.0 * tf.maximum(0.0, rotor_desnorm - 50.0)
                loss = -potencia_norm + penalty
            
            grads = tape.gradient(loss, [rotor_var])
            optimizer.apply_gradients(zip(grads, [rotor_var]))
            
            potencia_atual = float(potencia)
            rotor_atual = float(rotor_var * std['posicao_rotor'].iloc[0] + mean['posicao_rotor'].iloc[0])
            
            if potencia_atual > best_potencia and 0 <= rotor_atual <= 50:
                best_potencia = potencia_atual
                best_rotor = rotor_atual
        
        return best_rotor, best_potencia
    
    resultados_otimizacao = []

    # Limitar o número de amostras para otimização
    indices = np.random.choice(len(X_test), size=min(LIMITE_AMOSTRAS, len(X_test)), replace=False)
    X_test_limitado = X_test.iloc[indices]
    y_test_limitado = y_test[indices]

    print(f'LIMITE_AMOSTRAS: {LIMITE_AMOSTRAS}, indices: {len(indices)}')
    print('X_test_limitado.shape', X_test_limitado.shape)
    print('y_test_limitado.shape', y_test_limitado.shape)

    
    
    # Para cada lote do conjunto de teste limitado
    total_amostras = len(X_test_limitado)
    tempo_inicio = datetime.now()
    tempos_lote = []

    for idx in range(0, total_amostras, BATCH_SIZE):
        tempo_lote_inicio = datetime.now()
        
        # Pegar o lote atual
        end_idx = min(idx + BATCH_SIZE, total_amostras)
        x_batch = X_test_limitado.iloc[idx:end_idx]
        y_batch = y_test_limitado[idx:end_idx]

        print(x_batch.shape)
        print(y_batch.shape)

        # print(f'idx: {idx}, end_idx: {end_idx}')

        # Processar o lote
        batch_results = []
        for i in range(len(x_batch)):
            # start_time = datetime.now()
            x_sample = x_batch.iloc[i:i+1]
            rotor_atual = x_sample['posicao_rotor'].values[0]
            potencia_atual = y_batch[i]
            
            melhor_rotor, melhor_potencia = otimizar_rotor_tf(x_sample)

            # end_time = datetime.now()
            # print(f'Tempo de execução: {end_time - start_time} - {i}')
            
            batch_results.append({
                'Distribuidor': x_sample['distribuidor'].values[0] * std['distribuidor'].iloc[0] + mean['distribuidor'].iloc[0],
                'Rotor Atual': rotor_atual * std['posicao_rotor'].iloc[0] + mean['posicao_rotor'].iloc[0],
                'Rotor Otimizado': melhor_rotor,
                'Potência Atual': potencia_atual,
                'Potência Otimizada': melhor_potencia,
                'Ganho (%)': ((melhor_potencia - potencia_atual) / potencia_atual) * 100
            })
        
        resultados_otimizacao.extend(batch_results)
        # print(len(resultados_otimizacao))
        # raise Exception("Stop")
        
        # Calcular tempo estimado
        tempo_lote_fim = datetime.now()
        tempo_lote = (tempo_lote_fim - tempo_lote_inicio).total_seconds()
        tempos_lote.append(tempo_lote)
        
        # Calcular médias e estimativas
        tempo_medio_lote = np.mean(tempos_lote)
        lotes_restantes = (total_amostras - end_idx) / BATCH_SIZE
        tempo_restante_estimado = tempo_medio_lote * lotes_restantes
        
        # Calcular progresso
        progresso = (idx + BATCH_SIZE) / total_amostras * 100
        
        # Formatar tempo restante
        tempo_restante = str(timedelta(seconds=int(tempo_restante_estimado)))
        tempo_decorrido = str(timedelta(seconds=int((tempo_lote_fim - tempo_inicio).total_seconds())))
        
        # Imprimir status
        print(f"\rProgresso: {progresso:.1f}% | "
              f"Lote: {idx//BATCH_SIZE + 1}/{total_amostras//BATCH_SIZE + 1} | "
              f"Tempo por lote: {tempo_lote:.1f}s | "
              f"Tempo decorrido: {tempo_decorrido} | "
              f"Tempo restante estimado: {tempo_restante}")
        
        # Salvar resultados parciais a cada 1000 amostras
        if idx % 64 == 0:
            print(f"Salvando resultados parciais: {idx}")
            print(resultados_otimizacao[-1])
            df_parcial = pd.DataFrame(resultados_otimizacao)
            df_parcial.to_csv(f'C:/projetos/engesep/5_dist_rotor/data/otimizacao_parcial_{idx}.csv', index=False)

    print("\nOtimização concluída!")
    tempo_total = str(timedelta(seconds=int((datetime.now() - tempo_inicio).total_seconds())))
    print(f"Tempo total de execução: {tempo_total}")

    # Converter resultados finais para DataFrame
    df_otimizacao = pd.DataFrame(resultados_otimizacao)

    # Salvar resultados finais
    df_otimizacao.to_csv('C:/projetos/engesep/5_dist_rotor/data/otimizacao_final.csv', index=False)

    # Mostrar estatísticas da otimização
    print("\nEstatísticas da Otimização:")
    print(f"Ganho Médio: {df_otimizacao['Ganho (%)'].mean():.2f}%")
    print(f"Ganho Máximo: {df_otimizacao['Ganho (%)'].max():.2f}%")
    print(f"Ganho Mínimo: {df_otimizacao['Ganho (%)'].min():.2f}%")
    print(f"Soma das potencias: {df_otimizacao['Potência Atual'].sum()/len(df_otimizacao):.2f}")
    print(f"Soma das potencias otimizadas: {df_otimizacao['Potência Otimizada'].sum()/len(df_otimizacao):.2f}")
    
    # Plotar resultados da otimização
    plt.figure(figsize=(12, 6))
    plt.scatter(df_otimizacao['Distribuidor'], df_otimizacao['Rotor Atual'], 
                alpha=0.5, label='Posição Atual')
    plt.scatter(df_otimizacao['Distribuidor'], df_otimizacao['Rotor Otimizado'], 
                alpha=0.5, label='Posição Otimizada')
    plt.title('Comparação: Posições Atuais vs Otimizadas do Rotor')
    plt.xlabel('Abertura do Distribuidor (%)')
    plt.ylabel('Posição do Rotor (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'C:\projetos\engesep\5_dist_rotor\data\otimizacao_rotor.png')
    plt.close()

    # encontrar a equação gerada pelas colunas distribuidor e Rotor Otimizado para df df_otimizacao
    # aplicar a regressão linear para encontrar a equação
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    # Reformatar os dados para o formato correto (2D array)
    X_reg = df_otimizacao['Distribuidor'].values.reshape(-1, 1)
    y_reg = df_otimizacao['Rotor Otimizado'].values
    
    # Treinar o modelo
    model.fit(X_reg, y_reg)

    # encontrar os pontos para o distribuidor 
    distribuidor_rotor = interpolate_data()
    X_pred = distribuidor_rotor['distribuidor'].values.reshape(-1, 1)
    rotor_otimizado = model.predict(X_pred)

    # merge do dataframe distribuidor_rotor com rotor_otimizado
    distribuidor_rotor['rotor_otimizado'] = rotor_otimizado

    # Imprimir os resultados
    print("\nComparação dos valores:")
    print("Distribuidor | Rotor Atual | Rotor Otimizado")
    print("-" * 45)
    for i in range(len(distribuidor_rotor)):
        print(f"{distribuidor_rotor['distribuidor'][i]:11.2f} | "
              f"{distribuidor_rotor['posicao_rotor'][i]:10.2f} | "
              f"{distribuidor_rotor['rotor_otimizado'][i]:14.2f}")

    # plotar o grafico do distribuidor no eixo x, rotor_otimizado e posicao_rotor no eixo y
    plt.figure(figsize=(10, 10))
    plt.scatter(distribuidor_rotor['distribuidor'], distribuidor_rotor['rotor_otimizado'], 
                label='Rotor Otimizado', alpha=0.7)
    plt.scatter(distribuidor_rotor['distribuidor'], distribuidor_rotor['posicao_rotor'], 
                label='Posição do Rotor', alpha=0.7)
    plt.title('Comparação: Posições Atuais vs Otimizadas do Rotor')
    plt.xlabel('Abertura do Distribuidor (%)')
    plt.ylabel('Posição do Rotor (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'C:\projetos\engesep\5_dist_rotor\data\otimizacao_rotor_final.png')
    plt.close()

    # model1 = np.polyfit(df_otimizacao['Distribuidor'], df_otimizacao['Rotor Otimizado'], 2)




# Primeiras 10 predições:
#     Valor Real  Valor Predito  Diferença
# 0       1202.0    1214.258057 -12.258057
# 1       1652.0    1609.306396  42.693604
# 2       1445.0    1437.517334   7.482666
# 3       1224.0    1217.390137   6.609863
# 4       1979.0    1973.362549   5.637451
# ..         ...            ...        ...
# 95      1184.0    1216.054565 -32.054565
# 96      1197.0    1202.482178  -5.482178
# 97      1534.0    1563.756958 -29.756958
# 98      1272.0    1285.343872 -13.343872
# 99       229.0     189.657104  39.342896

# [100 rows x 3 columns]

# Métricas de Erro:
# MAE: 17.18
# RMSE: 94.34
