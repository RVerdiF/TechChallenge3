import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
from datetime import datetime, timedelta
from pathlib import Path
import src.ApiHandler.data_api as data_api
import src.DataHandler.data_handler as data_handler
import src.DataHandler.feature_engineering as feature_engineering
import src.ModelHandler.predict as predict
import src.ModelHandler.train_model as train_model

st.set_page_config(page_title="Dashboard BTC", layout="wide")

DB_PATH = Path("src/DataHandler/btc_prices.db")
MODEL_PATH = Path("src/ModelHandler/lgbm_model.pkl")
METRICS_PATH = Path("src/ModelHandler/metrics.json")

def dashboard_page():
    st.title("Dashboard de Previsão de Preço do BTC")

    if st.button("Atualizar Dados"):
        with st.spinner("Atualizando dados..."):
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
            df = data_api.get_btc_data(start_date=start_date, end_date=end_date)
            if not df.empty:
                data_handler.save_data(df)
                st.success("Dados atualizados!")
            else:
                st.error("Erro ao atualizar dados")

    df = data_handler.load_data()
    if df.empty:
        st.warning("Nenhum dado encontrado. Clique em 'Atualizar Dados' para começar.")
        return

    # Carrega parâmetros de features do último treino
    feature_params = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            feature_params = metrics.get("feature_params", {})

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
                prediction, confidence = predict.make_prediction(df_features)
                if prediction == 1:
                    st.success(f"### Tendência para amanhã: **ALTA** 📈 (Confiança: {confidence:.2%})")
                else:
                    st.error(f"### Tendência para amanhã: **QUEDA** 📉 (Confiança: {confidence:.2%})")
        except FileNotFoundError:
            st.error("Modelo não encontrado. Execute o treinamento primeiro.")
        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")

    st.subheader("Histórico de Preços do Bitcoin")
    
    # Filtro de período
    period = st.radio("Selecione o período", ["1 mês", "3 meses", "1 ano", "Tudo"], index=3, horizontal=True)

    # Filtra o dataframe
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
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
                st.write("**Indicadores do Modelo:**", ", ".join(metrics.get("features", [])))

def settings_page():
    st.title("Configurações")

    st.header("Parâmetros de Treinamento do Modelo")
    n_estimators = st.number_input("Número de Estimadores", min_value=1, value=100, key="n_estimators")
    learning_rate = st.number_input("Taxa de Aprendizagem", min_value=0.01, value=0.1, step=0.01, key="learning_rate")

    st.header("Parâmetros dos Indicadores Técnicos")
    sma_short = st.number_input("Média Móvel Curta (SMA)", min_value=1, value=10, key="sma_short")
    sma_long = st.number_input("Média Móvel Longa (SMA)", min_value=1, value=30, key="sma_long")
    rsi_period = st.number_input("Período do RSI", min_value=1, value=14, key="rsi_period")

    if st.button("Treinar Modelo"):
        with st.spinner("Treinando modelo..."):
            try:
                model_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate
                }
                feature_params = {
                    'sma_window_1': sma_short,
                    'sma_window_2': sma_long,
                    'rsi_window': rsi_period
                }
                train_model.train_and_save_model(feature_params=feature_params, model_params=model_params)
                st.success("Modelo treinado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao treinar: {e}")

    st.subheader("Métricas do Modelo")
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Acurácia", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
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

def main():
    if not DB_PATH.exists() or not MODEL_PATH.exists():
        st.info("Primeira execução detectada. Preparando o ambiente...")
        with st.spinner("Atualizando dados e treinando o modelo. Isso pode levar alguns minutos..."):
            try:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
                df_new = data_api.get_btc_data(start_date=start_date, end_date=end_date)
                if not df_new.empty:
                    data_handler.save_data(df_new)
                    st.success("Dados atualizados com sucesso!")
                else:
                    st.error("Falha ao buscar novos dados.")
                    return
                train_model.train_and_save_model()
                st.success("Modelo treinado com sucesso!")
                st.balloons()
            except Exception as e:
                st.error(f"Ocorreu um erro durante a configuração inicial: {e}")
                return

    st.sidebar.title("Menu")
    page = st.sidebar.radio("Selecione uma página", ["Dashboard", "Configurações"])

    if page == "Dashboard":
        dashboard_page()
    elif page == "Configurações":
        settings_page()

if __name__ == "__main__":
    main()
