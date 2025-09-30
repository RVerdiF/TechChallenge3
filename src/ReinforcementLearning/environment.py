
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.LogHandler.log_config import get_logger

logger = get_logger(__name__)

INITIAL_ACCOUNT_BALANCE = 10000

class TradingEnv(gym.Env):
    """
    Um ambiente de negociação de ações para Aprendizado por Reforço.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, feature_cols: list, transaction_cost_pct: float = 0.001, holding_penalty: float = -0.01, trade_completion_bonus: float = 0.1):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index() # Mantém a data para o log de trades
        self.feature_cols = feature_cols
        self.initial_balance = INITIAL_ACCOUNT_BALANCE
        self.transaction_cost_pct = transaction_cost_pct
        self.holding_penalty = holding_penalty
        self.trade_completion_bonus = trade_completion_bonus
        
        # Espaço de observação com bounds realistas para melhor estabilidade
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(len(self.feature_cols) + 2,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)

    def _next_observation(self):
        """
        Obtém a próxima observação a ser fornecida ao agente.
        """
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        features = self.df.iloc[self.current_step][self.feature_cols].values
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Normaliza balance e shares_held para melhor aprendizado
        normalized_balance = self.balance / self.initial_balance
        normalized_shares = (self.shares_held * current_price) / self.initial_balance
        
        obs = np.append(features, [normalized_balance, normalized_shares])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para um estado inicial.
        """
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.position = None # None ou 'long'
        self.trades = []
        
        # Define o ponto de partida sempre como o início do dataframe fornecido
        self.current_step = 0
        
        observation = self._next_observation()
        info = {} 
        
        return observation, info

    def step(self, action):
        """
        Executa um passo no ambiente.
        """
        current_price = self.df.loc[self.current_step, 'Close']
        current_date = self.df.loc[self.current_step, 'date']
        prev_net_worth = self.net_worth
        
        is_inaction = False
        reward_bonus = 0

        # Ação de Comprar
        if action == 1:
            if self.balance > 0:
                shares_to_buy = (self.balance / current_price) * (1 - self.transaction_cost_pct)
                self.shares_held += shares_to_buy
                self.balance = 0
                self.position = 'long'
                self.trades.append({
                    'date_buy': current_date,
                    'price_buy': current_price,
                    'date_sell': None,
                    'price_sell': None,
                    'profit': None
                })
            else:
                is_inaction = True # Tentou comprar sem saldo

        # Ação de Vender
        elif action == 2:
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.balance *= (1 - self.transaction_cost_pct) # Aplica custo na venda
                self.shares_held = 0
                self.position = None
                reward_bonus += self.trade_completion_bonus # Adiciona bônus por fechar a posição
                if self.trades and self.trades[-1]['date_sell'] is None:
                    last_trade = self.trades[-1]
                    last_trade['date_sell'] = current_date
                    last_trade['price_sell'] = current_price
                    # Corrige cálculo do lucro baseado nas ações negociadas
                    shares_traded = (self.initial_balance / last_trade['price_buy']) * (1 - self.transaction_cost_pct)
                    last_trade['profit'] = (last_trade['price_sell'] - last_trade['price_buy']) * shares_traded
            else:
                is_inaction = True # Tentou vender sem ações
        
        # Ação de Manter
        else: # action == 0
            is_inaction = True

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        # Se o episódio terminar e ainda houver uma posição aberta, força a venda
        if terminated and self.position == 'long':
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.balance *= (1 - self.transaction_cost_pct)
                self.shares_held = 0
                self.position = None
                if self.trades and self.trades[-1]['date_sell'] is None:
                    last_trade = self.trades[-1]
                    last_trade['date_sell'] = current_date
                    last_trade['price_sell'] = current_price
                    # Corrige cálculo do lucro baseado nas ações negociadas
                    shares_traded = (self.initial_balance / last_trade['price_buy']) * (1 - self.transaction_cost_pct)
                    last_trade['profit'] = (last_trade['price_sell'] - last_trade['price_buy']) * shares_traded
        
        # Calcula a recompensa baseada no lucro/prejuízo
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Recompensa baseada na mudança percentual do patrimônio
        if prev_net_worth > 0:
            reward = ((self.net_worth - prev_net_worth) / prev_net_worth) * 100
        else:
            reward = 0
        
        # Adiciona bônus por trade completo
        reward += reward_bonus
        
        # Penalidade por inação para incentivar ações
        if is_inaction:
            reward += self.holding_penalty
        else:
            # Pequeno incentivo para tomar ações (comprar ou vender)
            reward += 0.01

        observation = self._next_observation()
        truncated = False 
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human', close=False):
        """
        Renderiza o estado atual do ambiente (opcional).
        """
        profit = self.net_worth - self.initial_balance
        logger.info(f'Passo: {self.current_step}')
        logger.info(f'Patrimônio Líquido: {self.net_worth:.2f}')
        logger.info(f'Lucro: {profit:.2f}')
