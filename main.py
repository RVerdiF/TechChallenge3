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
    if st.button("Gerar Previsão", type="primary"):
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
    st.title("Configurações")
    
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)
    
    config = model_db_handler.load_config(username)
    model_params_saved = config.get("model_params", {})
    feature_params_saved = config.get("feature_params", {})

    with st.expander("Parâmetros de Treinamento do Modelo", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.number_input("Número de Estimadores", min_value=1, value=model_params_saved.get('n_estimators', 100), key="n_estimators", help="Número de árvores de decisão no modelo. Mais árvores podem melhorar a precisão, mas aumentam o tempo de treinamento.")
            max_depth = st.number_input("Profundidade Máxima", min_value=-1, value=model_params_saved.get('max_depth', -1), key="max_depth", help="Profundidade máxima de cada árvore. -1 significa sem limite.")
            reg_alpha = st.number_input("Regularização L1 (Alpha)", min_value=0.0, value=model_params_saved.get('reg_alpha', 0.0), step=0.01, key="reg_alpha", help="Termo de regularização L1. Ajuda a prevenir overfitting.")
        with col2:
            learning_rate = st.number_input("Taxa de Aprendizagem", min_value=0.01, value=model_params_saved.get('learning_rate', 0.1), step=0.01, key="learning_rate", help="Taxa de aprendizagem. Um valor menor torna o aprendizado mais robusto, mas requer mais árvores.")
            num_leaves = st.number_input("Número de Folhas", min_value=2, value=model_params_saved.get('num_leaves', 31), key="num_leaves", help="Número máximo de folhas em uma árvore.")
            reg_lambda = st.number_input("Regularização L2 (Lambda)", min_value=0.0, value=model_params_saved.get('reg_lambda', 0.0), step=0.01, key="reg_lambda", help="Termo de regularização L2. Também ajuda a prevenir overfitting.")

    with st.expander("Parâmetros dos Indicadores Técnicos", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Médias Móveis")
            sma_short = st.number_input("SMA Curta", min_value=1, value=feature_params_saved.get('sma_window_1', 10), key="sma_short")
            sma_long = st.number_input("SMA Longa", min_value=1, value=feature_params_saved.get('sma_window_2', 30), key="sma_long")
            ema_short = st.number_input("EMA Curta", min_value=1, value=feature_params_saved.get('ema_window_1', 12), key="ema_short")
            ema_long = st.number_input("EMA Longa", min_value=1, value=feature_params_saved.get('ema_window_2', 26), key="ema_long")
        with col2:
            st.subheader("Osciladores")
            rsi_period = st.number_input("Período RSI", min_value=1, value=feature_params_saved.get('rsi_window', 14), key="rsi_period")
            stochastic_window = st.number_input("Janela Estocástico", min_value=1, value=feature_params_saved.get('stochastic_window', 14), key="stochastic_window")
            bollinger_window = st.number_input("Janela Bollinger", min_value=1, value=feature_params_saved.get('bollinger_window', 20), key="bollinger_window")
        with col3:
            st.subheader("MACD")
            macd_fast = st.number_input("MACD Rápido", min_value=1, value=feature_params_saved.get('macd_fast', 12), key="macd_fast")
            macd_slow = st.number_input("MACD Lento", min_value=1, value=feature_params_saved.get('macd_slow', 26), key="macd_slow")
            macd_signal = st.number_input("MACD Sinal", min_value=1, value=feature_params_saved.get('macd_signal', 9), key="macd_signal")

    st.subheader("Período de Treinamento")
    col1, col2 = st.columns(2)
    with col1:
        train_start_date = st.date_input("Data de Início do Treinamento", value=min_date)
    with col2:
        train_end_date = st.date_input("Data de Fim do Treinamento", value=datetime.now())

    if st.button("Treinar Modelo"):
        logger.info(f"User '{username}' clicked 'Train Model'.")
        with st.spinner("Treinando modelo..."):
            try:
                model_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'num_leaves': num_leaves,
                    'reg_alpha': reg_alpha,
                    'reg_lambda': reg_lambda
                }
                feature_params = {
                    'sma_window_1': sma_short,
                    'sma_window_2': sma_long,
                    'rsi_window': rsi_period,
                    'ema_window_1': ema_short,
                    'ema_window_2': ema_long,
                    'macd_fast': macd_fast,
                    'macd_slow': macd_slow,
                    'macd_signal': macd_signal,
                    'bollinger_window': bollinger_window,
                    'stochastic_window': stochastic_window
                }
                config_to_save = {"model_params": model_params, "feature_params": feature_params}
                model_db_handler.save_config(username, config_to_save)
                
                train_model.train_and_save_model(
                    username=username,
                    feature_params=feature_params, 
                    model_params=model_params, 
                    start_date=train_start_date,
                    end_date=train_end_date
                )
                st.success("Modelo treinado com sucesso!")
            except Exception as e:
                logger.error(f"Error during training for user '{username}': {e}", exc_info=True)
                st.error(f"Erro durante o treinamento: {e}")

    _, metrics = model_db_handler.load_model(username)
    if metrics:
        st.subheader("Métricas do Modelo Atual")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Acurácia", f"{metrics.get('accuracy', 0):.2%}")
        with col2:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
        with st.expander("Matriz de Confusão"):
            conf_matrix = metrics.get("confusion_matrix")
            if conf_matrix:
                z = conf_matrix
                x = ['Queda', 'Alta']
                y = ['Queda', 'Alta']
                fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
                fig_cm.update_layout(title='Matriz de Confusão')
                st.plotly_chart(fig_cm, use_container_width=True, key="confusion_matrix_chart")
    else:
        st.warning("Métricas não encontradas. Treine o modelo para gerá-las.")

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
                
                st.line_chart(chart_data)

                st.subheader("Histórico de Trades")
                st.dataframe(trades_history)

            except Exception as e:
                logger.error(f"An error occurred during backtest for user '{username}': {e}", exc_info=True)
                st.error(f"Ocorreu um erro durante o backtest: {e}")

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
        page = st.sidebar.radio("Selecione uma página", ["Dashboard", "Configurações", "Backtesting"])

        logger.info(f"User '{username}' navigated to page: {page}")
        if page == "Dashboard":
            dashboard_page(username)
        elif page == "Settings":
            settings_page(username)
        elif page == "Backtesting":
            backtesting_page(username)
        
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