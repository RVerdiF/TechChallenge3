
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

INITIAL_ACCOUNT_BALANCE = 10000
LOOKBACK_WINDOW_SIZE = 30 # Observa os últimos 30 dias de preço

class TradingEnv(gym.Env):
    """
    Um ambiente de negociação de ações para Aprendizado por Reforço.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame):
        super(TradingEnv, self).__init__()

        self.df = df
        self.current_step = 0
        self.initial_balance = INITIAL_ACCOUNT_BALANCE
        
        # Definir o espaço de ação: 0: Manter, 1: Comprar, 2: Vender
        self.action_space = spaces.Discrete(3)

        # Definir o espaço de observação (estado)
        # Consiste em: [Preços dos últimos N dias] + [saldo, ações em posse]
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(LOOKBACK_WINDOW_SIZE + 2,), 
            dtype=np.float32
        )

    def _next_observation(self):
        """
        Obtém a próxima observação a ser fornecida ao agente.
        """
        # Obtém a janela de dados de preços
        frame = self.df.loc[
            self.current_step - LOOKBACK_WINDOW_SIZE + 1 : self.current_step, 'Close'
        ].values
        
        # Adiciona o saldo e as ações em posse
        obs = np.append(frame, [self.balance, self.shares_held])
        
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para um estado inicial.
        """
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        
        # Define o ponto de partida aleatório no dataframe
        self.current_step = np.random.randint(
            LOOKBACK_WINDOW_SIZE, len(self.df) - 1
        )
        
        observation = self._next_observation()
        info = {} # Dicionário para informações de debug
        
        return observation, info

    def step(self, action):
        """
        Executa um passo no ambiente.
        """
        current_price = self.df.loc[self.current_step, 'Close']
        
        # Executa a ação
        if action == 1: # Comprar
            # Compra o máximo possível com o saldo
            if self.balance > 0:
                shares_to_buy = self.balance / current_price
                self.shares_held += shares_to_buy
                self.balance = 0

        elif action == 2: # Vender
            # Vende todas as ações
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

        # Calcula o novo patrimônio líquido
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calcula a recompensa
        reward = self.net_worth - prev_net_worth
        
        # Avança no tempo
        self.current_step += 1
        
        # Verifica se o episódio terminou
        terminated = self.current_step >= len(self.df) - 1
        
        # Obtém a próxima observação
        observation = self._next_observation()
        
        # O Gymnasium espera 5 valores de retorno
        truncated = False 
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human', close=False):
        """
        Renderiza o estado atual do ambiente (opcional).
        """
        profit = self.net_worth - self.initial_balance
        print(f'Passo: {self.current_step}')
        print(f'Patrimônio Líquido: {self.net_worth:.2f}')
        print(f'Lucro: {profit:.2f}')
