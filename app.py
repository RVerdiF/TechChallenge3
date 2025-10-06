import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
import threading
from src.health_check import run_health_check_server
from src.LogHandler.log_config import get_logger
from datetime import datetime, date

import src.ApiHandler.data_api as data_api
import src.DataHandler.data_handler as data_handler
import src.DataHandler.feature_engineering as feature_engineering
import src.ModelHandler.predict as predict
import src.ModelHandler.train_model as train_model
from src.AuthHandler import auth
from src.BacktestHandler import backtesting
import src.DataHandler.model_db_handler as model_db_handler
from src.Orchestration.update_scheduler import daily_update_task, LAST_UPDATE_KEY

# --- Background Tasks ---
if 'health_check_started' not in st.session_state:
    health_thread = threading.Thread(target=run_health_check_server, daemon=True)
    health_thread.start()
    st.session_state['health_check_started'] = True

if 'daily_update_task_started' not in st.session_state:
    update_thread = threading.Thread(target=daily_update_task, daemon=True)
    update_thread.start()
    st.session_state['daily_update_task_started'] = True
# -------------------------

st.set_page_config(page_title="BTC Dashboard", layout="wide")

logger = get_logger(__name__)

# --- Database Initialization ---
auth.init_db()
model_db_handler.init_db()
data_handler.init_database()
# -----------------------------

def dashboard_page(username):
    logger.info(f"Displaying dashboard page for user '{username}'.")
    st.title("BTC Price Prediction Dashboard")

    # Indicador de última atualização
    last_update_str = data_handler.get_metadata(LAST_UPDATE_KEY)
    if last_update_str:
        try:
            last_update_date = datetime.strptime(last_update_str, "%Y-%m-%d").date()
            st.caption(f"Última atualização dos dados: {last_update_date.strftime('%d/%m/%Y')}")
        except (ValueError, TypeError):
            st.caption("Aguardando registro de atualização de dados...")
    else:
        st.caption("Dados sendo atualizados pela primeira vez em segundo plano...")

    # Load data to get the minimum date
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)

    # Date controls for manual update
    with st.expander("Atualização Manual de Dados"):
        st.info("A atualização diária é automática. Use esta seção apenas para preencher ou corrigir um intervalo de datas específico.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        with col3:
            st.write("")
            st.write("")
            if st.button("Atualizar Intervalo"):
                logger.info(f"User '{username}' clicked 'Update Data Range'.")
                with st.spinner("Atualizando dados para o intervalo selecionado..."):
                    df_new = data_api.get_btc_data(
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d")
                    )
                    if not df_new.empty:
                        data_handler.update_data(df_new, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                        st.success("Dados atualizados para o intervalo selecionado!")
                        st.rerun() # Rerun to reflect changes
                    else:
                        st.error("Erro ao atualizar dados")

    df = data_handler.load_data()
    if df.empty:
        st.warning("Nenhum dado encontrado. A aplicação está buscando os dados iniciais em segundo plano. Por favor, aguarde um momento e atualize a página.")
        return

    config = model_db_handler.load_config(username)
    feature_params = config.get("feature_params", {})

    st.subheader("Visão Geral do Mercado")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Preço Atual", f"${df['Close'].iloc[-1]:,.2f}")
    with col2:
        change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        changepct = (change / df['Close'].iloc[-2]) * 100
        st.metric("Variação Diária", f"${change:,.2f}", f"%{changepct:+.2f}")
    with col3:
        st.metric("Máxima (30d)", f"${df['High'].tail(30).max():,.2f}")
    with col4:
        st.metric("Mínima (30d)", f"${df['Low'].tail(30).min():,.2f}")

    st.subheader("Previsão para o Próximo Dia")
    if st.button("Gerar Previsão (Modelo LightGBM)", type="primary"):
        logger.info(f"User '{username}' clicked 'Generate Tomorrow's Forecast'.")
        try:
            with st.spinner("Gerando previsão..."):
                model, metrics = model_db_handler.load_model(username)
                if model is None:
                    st.error("Modelo não encontrado. Por favor, treine um modelo primeiro na página de Configurações.")
                else:
                    loaded_feature_params = metrics.get("feature_params", feature_params)
                    df_features = feature_engineering.create_features(df, params=loaded_feature_params)
                    
                    if df_features.empty:
                        st.error("Dados insuficientes para gerar a previsão.")
                    else:
                        prediction, confidence = predict.make_prediction(df_features, model, metrics)
                        if prediction == 1:
                            st.success(f"### Tendência para amanhã: **ALTA** 📈 (Confiança: {confidence:.2%})")
                        else:
                            st.error(f"### Tendência para amanhã: **QUEDA** 📉 (Confiança: {confidence:.2%})")
        except Exception as e:
            logger.error(f"Error generating forecast for user '{username}': {e}", exc_info=True)
            st.error(f"Erro ao gerar previsão: {e}")

    st.subheader("Histórico de Preços do Bitcoin")
    
    period = st.radio("Selecionar período", ["1 mês", "3 meses", "1 ano", "Todos"], index=3, horizontal=True)

    if period != "Todos":
        end_date_dt = df.index.max()
        if period == "1 mês":
            start_date_dt = end_date_dt - pd.DateOffset(months=1)
        elif period == "3 meses":
            start_date_dt = end_date_dt - pd.DateOffset(months=3)
        else: # 1 ano
            start_date_dt = end_date_dt - pd.DateOffset(years=1)
        df_filtered = df[df.index >= start_date_dt]
    else:
        df_filtered = df

    fig = px.line(df_filtered.reset_index(), x='date', y='Close', title="Preço de Fechamento do BTC (USD)", labels={'date': 'Data', 'Close': 'Preço (USD)'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Informações do Dataset"):
        st.write(f"**Período dos dados:** {df.index.min().strftime('%d/%m/%Y')} a {df.index.max().strftime('%d/%m/%Y')}")
        st.write(f"**Total de registros:** {len(df)}")
        st.write("**Colunas:** Open, High, Low, Close, Volume")
        _, metrics = model_db_handler.load_model(username)
        if metrics:
            st.write("**Indicadores do Modelo:**", ", ".join(metrics.get("features", [])))

def settings_page(username):
    logger.info(f"Displaying settings page for user '{username}'.")
    st.title("Configurações e Treinamento de Modelos")

    # --- Carrega todas as configurações salvas ---
    config = model_db_handler.load_config(username)
    feature_params_saved = config.get("feature_params", {})
    model_params_saved = config.get("model_params", {})
    rl_params_saved = config.get("rl_params", {})

    # --- Seletor de Modelo ---
    selected_model = st.radio(
        "Selecione o Tipo de Modelo",
        ["LightGBM", "Aprendizado por Reforço"],
        horizontal=True,
        help="Escolha o modelo que deseja configurar e treinar."
    )
    st.divider()

    # --- Abas de Configuração ---
    tab_features, tab_hyperparams = st.tabs(["Parâmetros de Features", "Hiperparâmetros do Modelo"])

    with tab_features:
        st.subheader("Indicadores Técnicos para o Modelo")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Médias Móveis**")
            sma_short = st.number_input("SMA Curta", 1, value=feature_params_saved.get('sma_window_1', 10), key="sma_short")
            sma_long = st.number_input("SMA Longa", 1, value=feature_params_saved.get('sma_window_2', 30), key="sma_long")
            ema_short = st.number_input("EMA Curta", 1, value=feature_params_saved.get('ema_window_1', 12), key="ema_short")
            ema_long = st.number_input("EMA Longa", 1, value=feature_params_saved.get('ema_window_2', 26), key="ema_long")
        with col2:
            st.write("**Osciladores**")
            rsi_period = st.number_input("Período RSI", 1, value=feature_params_saved.get('rsi_window', 14), key="rsi_period")
            stochastic_window = st.number_input("Janela Estocástico", 1, value=feature_params_saved.get('stochastic_window', 14), key="stochastic_window")
            bollinger_window = st.number_input("Janela Bollinger", 1, value=feature_params_saved.get('bollinger_window', 20), key="bollinger_window")
        with col3:
            st.write("**MACD**")
            macd_fast = st.number_input("MACD Rápido", 1, value=feature_params_saved.get('macd_fast', 12), key="macd_fast")
            macd_slow = st.number_input("MACD Lento", 1, value=feature_params_saved.get('macd_slow', 26), key="macd_slow")
            macd_signal = st.number_input("MACD Sinal", 1, value=feature_params_saved.get('macd_signal', 9), key="macd_signal")

    with tab_hyperparams:
        if selected_model == "LightGBM":
            st.subheader("Hiperparâmetros do LightGBM")
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.number_input("Nº de Estimadores", 1, value=model_params_saved.get('n_estimators', 100), key="n_estimators")
                max_depth = st.number_input("Profundidade Máxima", -1, value=model_params_saved.get('max_depth', -1), key="max_depth")
                reg_alpha = st.number_input("Regularização L1", 0.0, value=model_params_saved.get('reg_alpha', 0.0), step=0.01, key="reg_alpha")
            with col2:
                learning_rate = st.number_input("Taxa de Aprendizagem", 0.01, value=model_params_saved.get('learning_rate', 0.1), step=0.01, key="learning_rate")
                num_leaves = st.number_input("Nº de Folhas", 2, value=model_params_saved.get('num_leaves', 31), key="num_leaves")
                reg_lambda = st.number_input("Regularização L2", 0.0, value=model_params_saved.get('reg_lambda', 0.0), step=0.01, key="reg_lambda")
        
        elif selected_model == "Aprendizado por Reforço":
            st.subheader("Hiperparâmetros do Agente PPO (RL)")
            col1, col2, col3 = st.columns(3)
            with col1:
                rl_total_timesteps = st.number_input("Total de Timesteps", 1000, value=rl_params_saved.get('total_timesteps', 50000), step=1000, key="rl_timesteps")
                rl_ent_coef = st.number_input("Coef. de Entropia", 0.0, 1.0, value=rl_params_saved.get('ent_coef', 0.1), step=0.001, format="%.3f", key="rl_ent_coef", help="Incentiva a exploração.")
            with col2:
                rl_learning_rate = st.number_input("Taxa de Aprendizagem", 1e-6, 1e-1, value=rl_params_saved.get('learning_rate', 0.0003), step=1e-5, format="%.4f", key="rl_lr")
                rl_gamma = st.number_input("Fator de Desconto (Gamma)", 0.2, 0.999, value=rl_params_saved.get('gamma', 0.99), step=0.001, key="rl_gamma")
            with col3:
                rl_transaction_cost = st.number_input("Custo de Transação (%)", 0.0, 100.0, value=rl_params_saved.get('transaction_cost_pct', 0.1) * 100, step=0.01, format="%.3f", key="rl_cost")
                rl_holding_penalty = st.number_input("Penalidade por Inação", -10.0, 0.0, value=rl_params_saved.get('holding_penalty', -0.1), step=0.01, format="%.3f", key="rl_penalty")
                rl_trade_bonus = st.number_input("Bônus por Trade Completo", 0.0, 100.0, value=rl_params_saved.get('trade_completion_bonus', 0.1), step=0.01, format="%.3f", key="rl_trade_bonus", help="Recompensa adicional por completar um trade.")

    st.divider()

    # --- Período de Treinamento ---
    st.subheader("Período de Treinamento")
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)
    col1, col2 = st.columns(2)
    with col1:
        train_start_date = st.date_input("Data de Início", value=min_date)
    with col2:
        train_end_date = st.date_input("Data de Fim", value=datetime.now().date())

    # --- Botão de Treinamento Unificado ---
    if st.button(f"Treinar Modelo {selected_model}", type="primary"):
        feature_params = {
            'sma_window_1': sma_short, 'sma_window_2': sma_long, 'rsi_window': rsi_period,
            'ema_window_1': ema_short, 'ema_window_2': ema_long, 'macd_fast': macd_fast,
            'macd_slow': macd_slow, 'macd_signal': macd_signal, 'bollinger_window': bollinger_window,
            'stochastic_window': stochastic_window
        }

        if selected_model == "LightGBM":
            with st.spinner("Treinando modelo LightGBM..."):
                try:
                    model_params = {
                        'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth,
                        'num_leaves': num_leaves, 'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda
                    }
                    # Salva a config do lgbm e preserva a config do RL já carregada
                    config_to_save = {"model_params": model_params, "feature_params": feature_params, "rl_params": rl_params_saved}
                    model_db_handler.save_config(username, config_to_save)
                    
                    train_model.train_and_save_model(username=username, feature_params=feature_params, model_params=model_params, start_date=train_start_date, end_date=train_end_date)
                    st.success("Modelo LightGBM treinado com sucesso!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error during LightGBM training: {e}", exc_info=True)
                    st.error(f"Erro durante o treinamento: {e}")

        elif selected_model == "Aprendizado por Reforço":
            with st.spinner("Treinando modelo de RL... Isso pode levar vários minutos."):
                try:
                    from src.ModelHandler.train_rl_model import train_rl_model
                    rl_params = {
                        'total_timesteps': rl_total_timesteps,
                        'learning_rate': rl_learning_rate,
                        'gamma': rl_gamma,
                        'ent_coef': rl_ent_coef,
                        'transaction_cost_pct': rl_transaction_cost / 100, # Converte de % para decimal
                        'holding_penalty': rl_holding_penalty,
                        'trade_completion_bonus': rl_trade_bonus
                    }
                    # Salva a config do RL e preserva a config do lgbm já carregada
                    config_to_save = {"model_params": model_params_saved, "feature_params": feature_params, "rl_params": rl_params}
                    model_db_handler.save_config(username, config_to_save)

                    train_rl_model(feature_params=feature_params, rl_params=rl_params, start_date=train_start_date, end_date=train_end_date)
                    st.success("Modelo de RL treinado com sucesso!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error during RL training: {e}", exc_info=True)
                    st.error(f"Erro durante o treinamento de RL: {e}")
    
    st.divider()

    # --- Exibição de Métricas Unificada ---
    st.subheader(f"Métricas do Modelo {selected_model} Ativo")
    if selected_model == "LightGBM":
        _, metrics = model_db_handler.load_model(username)
        if metrics:
            col1, col2 = st.columns(2)
            st.metric("Acurácia", f"{metrics.get('accuracy', 0):.2%}")
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
            with st.expander("Matriz de Confusão"):
                conf_matrix = metrics.get("confusion_matrix")
                if conf_matrix:
                    fig_cm = ff.create_annotated_heatmap(conf_matrix, x=['Queda', 'Alta'], y=['Queda', 'Alta'], colorscale='Blues')
                    st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Métricas não encontradas. Treine o modelo para vê-las.")

    elif selected_model == "Aprendizado por Reforço":
        _, _, metrics, _ = model_db_handler.load_rl_model()
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("Retorno Total", f"{metrics.get('total_return_pct', 0):.2f}%")
            col2.metric("Índice de Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
            col3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        else:
            st.warning("Métricas não encontradas. Treine o modelo de RL para vê-las.")

def backtesting_page(username):
    logger.info(f"Displaying backtesting page for user '{username}'.")
    st.title("Backtesting de Estratégia")

    model, metrics = model_db_handler.load_model(username)
    if not model or not metrics:
        st.warning("Modelo ou métricas não encontrados. Treine um modelo primeiro na página de Configurações.")
        return

    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)

    st.subheader("Período do Backtest")
    col1, col2 = st.columns(2)
    with col1:
        backtest_start_date = st.date_input("Data de Início do Backtest", value=min_date)
    with col2:
        backtest_end_date = st.date_input("Data de Fim do Backtest", value=datetime.now())

    if st.button("Iniciar Backtest", type="primary"):
        logger.info(f"User '{username}' clicked 'Start Backtest'.")
        with st.spinner("Executando backtest... Isso pode levar alguns minutos."):
            try:
                feature_cols = metrics.get("features")
                feature_params = metrics.get("feature_params")

                if not feature_cols or not feature_params:
                    st.error("Informações de features não encontradas nas métricas. Por favor, treine o modelo novamente.")
                    return

                df = data_handler.load_data()
                df_features = feature_engineering.create_features(df, params=feature_params)

                results, trades_history = backtesting.run_backtest(
                    df_features, 
                    model, 
                    feature_cols, 
                    start_date=backtest_start_date, 
                    end_date=backtest_end_date
                )

                st.subheader("Resultados do Backtest")
                col1, col2, col3 = st.columns(3)
                col1.metric("Retorno Total da Estratégia", f"{results['total_return_pct']:.2f}%")
                col1.metric("Retorno Buy & Hold", f"{results['buy_and_hold_return_pct']:.2f}%")
                col2.metric("Índice de Sharpe", f"{results['sharpe_ratio']:.2f}")
                col2.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                col3.metric("Total de Trades", results['total_trades'])
                col3.metric("Taxa de Acerto", f"{results['win_rate']:.2f}%")

                st.subheader("Evolução do Portfólio")
                portfolio_history = results['portfolio_history']['value']
                buy_and_hold_history = results['buy_and_hold_history']
                
                chart_data = pd.concat([portfolio_history, buy_and_hold_history], axis=1)
                chart_data.columns = ['Estratégia', 'Buy & Hold']
                
                fig = px.line(chart_data.reset_index(), x='date', y=['Estratégia', 'Buy & Hold'], title="Evolução do Portfólio vs Buy & Hold")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Histórico de Trades")
                st.dataframe(trades_history)

            except Exception as e:
                logger.error(f"An error occurred during backtest for user '{username}': {e}", exc_info=True)
                st.error(f"Ocorreu um erro durante o backtest: {e}")

def rl_backtesting_page(username):
    logger.info(f"Displaying RL backtesting page for user '{username}'.")
    st.title("Backtesting de Estratégia com RL")

    model, feature_params, metrics, rl_params = model_db_handler.load_rl_model()
    if model is None:
        st.warning("Modelo de RL não encontrado. Treine um modelo primeiro na página de Configurações.")
        return

    # st.info("Esta página executa uma simulação usando o último modelo de RL treinado para o período de datas selecionado.")
    
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)

    st.subheader("Período do Backtest")
    col1, col2 = st.columns(2)
    with col1:
        backtest_start_date = st.date_input("Data de Início do Backtest", value=min_date, key="rl_backtest_start")
    with col2:
        backtest_end_date = st.date_input("Data de Fim do Backtest", value=datetime.now().date(), key="rl_backtest_end")

    if st.button("Iniciar Backtest de RL", type="primary"):
        logger.info(f"User '{username}' clicked 'Start RL Backtest'.")
        with st.spinner("Executando backtest de RL..."):
            try:
                from src.ReinforcementLearning.environment import TradingEnv
                from src.BacktestHandler.rl_backtesting import run_rl_backtest

                # st.write("**Parâmetros de Features do Modelo Carregado:**")
                # st.json(feature_params)

                df_raw = data_handler.load_data()
                if df_raw.empty:
                    st.warning("Dados não encontrados.")
                    return
                
                df_period = df_raw.loc[backtest_start_date:backtest_end_date]

                df_features = feature_engineering.create_features(df_period, params=feature_params)
                feature_cols = feature_engineering.get_feature_names(params=feature_params)
                df_features.dropna(inplace=True)

                if df_features.empty:
                    st.error("Não há dados suficientes para o período selecionado após a criação das features.")
                    return

                env = TradingEnv(df_features, feature_cols=feature_cols)
                model.set_env(env)

                results = run_rl_backtest(model, env)

                st.subheader("Resultados do Backtest de RL")
                col1, col2, col3 = st.columns(3)
                col1.metric("Retorno Total da Estratégia", f"{results.get('total_return_pct', 0):.2f}%")
                col1.metric("Retorno Buy & Hold", f"{results.get('buy_and_hold_return_pct', 0):.2f}%")
                col2.metric("Índice de Sharpe", f"{results.get('sharpe_ratio', 0):.2f}")
                col2.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
                col3.metric("Total de Trades", f"{results.get('total_trades', 0)}")
                col3.metric("Taxa de Acerto", f"{results.get('win_rate', 0):.2f}%")

                st.subheader("Evolução do Portfólio")
                portfolio_history = results['portfolio_history']
                buy_and_hold_history = results['buy_and_hold_history']
                chart_data = pd.concat([portfolio_history, buy_and_hold_history], axis=1)
                chart_data.columns = ['Estratégia RL', 'Buy & Hold']
                
                fig = px.line(chart_data.reset_index(), x='date', y=['Estratégia RL', 'Buy & Hold'], title="Evolução do Portfólio (RL) vs Buy & Hold")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Histórico de Trades")
                st.dataframe(results['trades_history'])

            except Exception as e:
                logger.error(f"An error occurred during RL backtest for user '{username}': {e}", exc_info=True)
                st.error(f"Ocorreu um erro durante o backtest de RL: {e}")

def login_page():
    logger.info("Displaying login page.")
    st.title("Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Login"):
        logger.info(f"Login attempt for user '{username}'.")
        token = auth.login_user(username, password)
        if token:
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos")

    if st.button("Não tem uma conta? Cadastre-se"):
        logger.info("Navigating to registration page.")
        st.session_state['page'] = 'registration'
        st.rerun()

def registration_page():
    logger.info("Displaying registration page.")
    st.title("Cadastro")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Cadastrar"):
        logger.info(f"Registration attempt for user '{username}'.")
        if auth.create_user(username, password):
            st.success("Usuário criado com sucesso! Por favor, faça o login.")
            st.session_state['page'] = 'login'
            st.rerun()
        else:
            st.error("Usuário já existe")

    if st.button("Já tem uma conta? Faça o login"):
        logger.info("Navigating to login page.")
        st.session_state['page'] = 'login'
        st.rerun()

def main():
    if st.session_state.get('authentication_status', False):
        username = st.session_state['username']
        logger.info(f"User '{username}' is logged in.")
        
        st.sidebar.title(f"Bem-vindo, {username}")
        page = st.sidebar.radio("Selecione uma página", ["Dashboard", "Settings", "Backtesting", "Backtesting RL"])

        logger.info(f"User '{username}' navigated to page: {page}")
        if page == "Dashboard":
            dashboard_page(username)
        elif page == "Settings":
            settings_page(username)
        elif page == "Backtesting":
            backtesting_page(username)
        elif page == "Backtesting RL":
            rl_backtesting_page(username)
        
        if st.sidebar.button("Logout"):
            logger.info(f"User '{username}' logged out.")
            st.session_state['authentication_status'] = False
            st.session_state['username'] = None
            st.rerun()
    else:
        if st.session_state.get('page', 'login') == 'login':
            login_page()
        else:
            registration_page()

if __name__ == "__main__":
    main()