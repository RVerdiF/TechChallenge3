
import pandas as pd
from stable_baselines3 import PPO
from src.DataHandler.model_db_handler import load_data, save_rl_model
from src.ReinforcementLearning.environment import TradingEnv


def train_rl_model():
    """
    Carrega os dados, instancia o ambiente de RL e treina um agente PPO.
    """
    print("Iniciando o treinamento do modelo de RL...")

    # 1. Carregar dados
    print("Carregando dados do banco de dados...")
    df = load_data()
    if df.empty or len(df) < 100: # Validação mínima de dados
        print("Dados insuficientes para treinamento. Abortando.")
        return

    # 2. Instanciar o ambiente
    print("Instanciando o ambiente de negociação...")
    env = TradingEnv(df)

    # 3. Instanciar o agente PPO
    # 'MlpPolicy' é uma política de rede neural padrão.
    # verbose=1 para imprimir o progresso do treinamento.
    print("Instanciando o agente PPO...")
    model = PPO("MlpPolicy", env, verbose=1)

    # 4. Treinar o agente
    # total_timesteps é o número de passos de interação com o ambiente.
    # Um valor maior geralmente leva a um desempenho melhor, mas demora mais.
    print("Iniciando o loop de treinamento...")
    model.learn(total_timesteps=20000)

    # 5. Salvar o modelo no banco de dados
    print(f"Treinamento concluído. Salvando o modelo no banco de dados...")
    save_rl_model(model)
    print("Modelo salvo com sucesso!")

if __name__ == '__main__':
    train_rl_model()
