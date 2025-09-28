
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
import joblib
from datetime import datetime, timedelta, date
from pathlib import Path
import src.ApiHandler.data_api as data_api
import src.DataHandler.data_handler as data_handler
import src.DataHandler.feature_engineering as feature_engineering
import src.ModelHandler.predict as predict
import src.ModelHandler.train_model as train_model
from src.AuthHandler import auth
from src.BacktestHandler import backtesting

st.set_page_config(page_title="Dashboard BTC", layout="wide")

DB_PATH = Path("src/DataHandler/btc_prices.db")
CONFIG_PATH = Path("config.json")

auth.init_db()

def get_user_paths(username):
    user_dir = Path(f"user_data/{username}")
    model_path = user_dir / "lgbm_model.pkl"
    metrics_path = user_dir / "metrics.json"
    return model_path, metrics_path

def load_config():
    """Carrega a configuração do arquivo JSON."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    else:
        # Valores padrão
        return {
            "model_params": {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': -1,
                'num_leaves': 31,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0
            },
            "feature_params": {
                'sma_window_1': 10,
                'sma_window_2': 30,
                'rsi_window': 14,
                'ema_window_1': 12,
                'ema_window_2': 26,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bollinger_window': 20,
                'stochastic_window': 14
            }
        }

def save_config(config):
    """Salva a configuração em um arquivo JSON."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def dashboard_page(model_path, metrics_path):
    st.title("Dashboard de Previsão de Preço do BTC")

    # Carrega dados para obter data mínima
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)

    # Controles de data para atualização
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_date = st.date_input(
            "Data Inicial",
            value=min_date,
            max_value=datetime.now().date()
        )
    with col2:
        end_date = st.date_input(
            "Data Final",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
    with col3:
        st.write("")
        st.write("")
        if st.button("Atualizar Dados"):
            with st.spinner("Atualizando dados..."):
                data_handler.drop_table()
                df = data_api.get_btc_data(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                if not df.empty:
                    data_handler.save_data(df)
                    st.success("Dados atualizados!")
                else:
                    st.error("Erro ao atualizar dados")

    df = data_handler.load_data()
    if df.empty:
        st.warning("Nenhum dado encontrado. Clique em 'Atualizar Dados' para começar.")
        return

    config = load_config()
    feature_params = config.get("feature_params", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Preço Atual", f"${df['Close'].iloc[-1]:,.2f}")
    with col2:
        change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        changepct = (change / df['Close'].iloc[-2]) * 100
        st.metric("Variação Diária", f"${change:,.2f}", f"%{changepct:+.2f}")
    with col3:
        st.metric("Máximo (30d)", f"${df['High'].tail(30).max():,.2f}")
    with col4:
        st.metric("Mínimo (30d)", f"${df['Low'].tail(30).min():,.2f}")

    st.subheader("Previsão para Amanhã")
    if st.button("Gerar Previsão para Amanhã", type="primary"):
        try:
            with st.spinner("Gerando previsão..."):
                df_features = feature_engineering.create_features(df, params=feature_params)
                if df_features.empty:
                    st.error("Dados insuficientes para gerar previsão")
                    return
                prediction, confidence = predict.make_prediction(df_features, model_path)
                if prediction == 1:
                    st.success(f"### Tendência para amanhã: **ALTA** 📈 (Confiança: {confidence:.2%})")
                else:
                    st.error(f"### Tendência para amanhã: **QUEDA** 📉 (Confiança: {confidence:.2%})")
        except FileNotFoundError:
            st.error("Modelo não encontrado. Execute o treinamento primeiro.")
        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")

    st.subheader("Histórico de Preços do Bitcoin")
    
    period = st.radio("Selecione o período", ["1 mês", "3 meses", "1 ano", "Tudo"], index=3, horizontal=True)

    if period != "Tudo":
        end_date = df.index.max()
        if period == "1 mês":
            start_date = end_date - pd.DateOffset(months=1)
        elif period == "3 meses":
            start_date = end_date - pd.DateOffset(months=3)
        else: # 1 ano
            start_date = end_date - pd.DateOffset(years=1)
        df_filtered = df[df.index >= start_date]
    else:
        df_filtered = df

    fig = px.line(df_filtered.reset_index(), x='date', y='Close', title="Preço de Fechamento do BTC (USD)", labels={'date': 'Data', 'Close': 'Preço (USD)'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Informações do Dataset"):
        st.write(f"**Período dos dados:** {df.index.min().strftime('%Y-%m-%d')} até {df.index.max().strftime('%Y-%m-%d')}")
        st.write(f"**Total de registros:** {len(df)}")
        st.write("**Colunas:** Open, High, Low, Close, Volume")
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                st.write("**Indicadores do Modelo:**", ", ".join(metrics.get("features", [])))

def settings_page(model_path, metrics_path):
    st.title("Configurações")
    
    # Carrega dados para obter data mínima
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)
    
    config = load_config()
    model_params_saved = config.get("model_params", {})
    feature_params_saved = config.get("feature_params", {})

    with st.expander("Parâmetros de Treinamento do Modelo", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.number_input("Número de Estimadores", min_value=1, value=model_params_saved.get('n_estimators', 100), key="n_estimators", help="Número de árvores de decisão no modelo. Mais árvores podem melhorar a precisão, mas também aumentam o tempo de treinamento.")
            max_depth = st.number_input("Profundidade Máxima", min_value=-1, value=model_params_saved.get('max_depth', -1), key="max_depth", help="Profundidade máxima de cada árvore. Um valor maior pode capturar padrões mais complexos, mas também pode levar a overfitting. -1 significa sem limite.")
            reg_alpha = st.number_input("Regularização L1 (Alpha)", min_value=0.0, value=model_params_saved.get('reg_alpha', 0.0), step=0.01, key="reg_alpha", help="Termo de regularização L1. Ajuda a prevenir overfitting, penalizando pesos grandes. Útil quando há muitas features.")
        with col2:
            learning_rate = st.number_input("Taxa de Aprendizagem", min_value=0.01, value=model_params_saved.get('learning_rate', 0.1), step=0.01, key="learning_rate", help="Taxa de aprendizado. Um valor menor torna o aprendizado mais robusto, mas exige mais árvores (n_estimators).")
            num_leaves = st.number_input("Número de Folhas", min_value=2, value=model_params_saved.get('num_leaves', 31), key="num_leaves", help="Número máximo de folhas em uma árvore. Um valor maior aumenta a complexidade do modelo.")
            reg_lambda = st.number_input("Regularização L2 (Lambda)", min_value=0.0, value=model_params_saved.get('reg_lambda', 0.0), step=0.01, key="reg_lambda", help="Termo de regularização L2. Também ajuda a prevenir overfitting, de forma semelhante ao L1.")

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
                save_config({"model_params": model_params, "feature_params": feature_params})
                
                train_model.train_and_save_model(
                    feature_params=feature_params, 
                    model_params=model_params, 
                    model_path=model_path, 
                    metrics_path=metrics_path,
                    start_date=train_start_date,
                    end_date=train_end_date
                )
                st.success("Modelo treinado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao treinar: {e}")

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        if metrics:
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
    else:
        st.warning("Métricas não encontradas. Treine o modelo para gerá-las.")

def backtesting_page(model_path, metrics_path):
    st.title("Backtesting de Estratégia")

    if not model_path.exists() or not metrics_path.exists():
        st.warning("Modelo ou métricas não encontrados. Treine um modelo primeiro na página de Configurações.")
        return

    # Carrega dados para obter data mínima
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)

    st.subheader("Período do Backtest")
    col1, col2 = st.columns(2)
    with col1:
        backtest_start_date = st.date_input("Data de Início do Backtest", value=min_date)
    with col2:
        backtest_end_date = st.date_input("Data de Fim do Backtest", value=datetime.now())

    if st.button("Iniciar Backtest", type="primary"):
        with st.spinner("Executando backtest... Isso pode levar alguns minutos."):
            try:
                model = joblib.load(model_path)
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                
                feature_cols = metrics.get("features")
                feature_params = metrics.get("feature_params")

                if not feature_cols or not feature_params:
                    st.error("Informações de features não encontradas nas métricas. Treine o modelo novamente.")
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
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retorno Total da Estratégia", f"{results['total_return_pct']:.2f}%")
                col2.metric("Retorno Buy & Hold", f"{results['buy_and_hold_return_pct']:.2f}%")
                col3.metric("Total de Operações", results['total_trades'])
                col4.metric("Taxa de Acerto (Win Rate)", f"{results['win_rate']:.2f}%")

                st.subheader("Evolução do Portfólio")
                st.line_chart(results['portfolio_history'])

                st.subheader("Histórico de Operações")
                st.dataframe(trades_history)

            except Exception as e:
                st.error(f"Ocorreu um erro durante o backtest: {e}")

def login_page():
    st.title("Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Login"):
        token = auth.login_user(username, password)
        if token:
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos")

    if st.button("Não tem uma conta? Cadastre-se"):
        st.session_state['page'] = 'registration'
        st.rerun()

def registration_page():
    st.title("Cadastro")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Cadastrar"):
        if auth.create_user(username, password):
            st.success("Usuário criado com sucesso! Faça o login.")
            st.session_state['page'] = 'login'
            st.rerun()
        else:
            st.error("Usuário já existe")

    if st.button("Já tem uma conta? Faça o login"):
        st.session_state['page'] = 'login'
        st.rerun()

def main():
    if st.session_state.get('authentication_status', False):
        username = st.session_state['username']
        model_path, metrics_path = get_user_paths(username)

        st.sidebar.title(f"Bem-vindo, {username}")
        page = st.sidebar.radio("Selecione uma página", ["Dashboard", "Configurações", "Backtesting"])

        if page == "Dashboard":
            dashboard_page(model_path, metrics_path)
        elif page == "Configurações":
            settings_page(model_path, metrics_path)
        elif page == "Backtesting":
            backtesting_page(model_path, metrics_path)
        
        if st.sidebar.button("Logout"):
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
