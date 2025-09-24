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

def main():
    st.title("Dashboard de Previsão de Preço do BTC")

    # Verifica se é a primeira execução
    if not DB_PATH.exists() or not MODEL_PATH.exists():
        st.info("Primeira execução detectada. Preparando o ambiente...")
        with st.spinner("Atualizando dados e treinando o modelo. Isso pode levar alguns minutos..."):
            try:
                # Atualiza dados
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
                df_new = data_api.get_btc_data(start_date=start_date, end_date=end_date)
                if not df_new.empty:
                    data_handler.save_data(df_new)
                    st.success("Dados atualizados com sucesso!")
                else:
                    st.error("Falha ao buscar novos dados.")
                    return

                # Treina o modelo
                train_model.train_and_save_model()
                st.success("Modelo treinado com sucesso!")
                st.balloons()
            except Exception as e:
                st.error(f"Ocorreu um erro durante a configuração inicial: {e}")
                return

    # Sidebar para controles
    st.sidebar.header("Controles")
    
    # Botão para atualizar dados
    if st.sidebar.button("Atualizar Dados"):
        with st.spinner("Atualizando dados..."):
            # Busca dados dos últimos 1095 dias
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
            
            df = data_api.get_btc_data(start_date=start_date, end_date=end_date)
            if not df.empty:
                data_handler.save_data(df)
                st.sidebar.success("Dados atualizados!")
            else:
                st.sidebar.error("Erro ao atualizar dados")

    # Botão para treinar modelo
    if st.sidebar.button("Atualizar Modelo"):
        with st.spinner("Treinando modelo..."):
            try:
                train_model.train_and_save_model()
                st.sidebar.success("Modelo treinado com sucesso!")
            except Exception as e:
                st.sidebar.error(f"Erro ao treinar: {e}")
    
    # Carrega dados existentes
    df = data_handler.load_data()
    
    if df.empty:
        st.warning("Nenhum dado encontrado. Clique em 'Atualizar Dados' para começar.")
        return
    
    # Estatísticas básicas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Preço Atual", f"${df['Close'].iloc[-1]:,.2f}")
    
    with col2:
        change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        st.metric("Variação Diária", f"${change:,.2f}", f"{change:+.2f}")
    
    with col3:
        st.metric("Máximo (30d)", f"${df['High'].tail(30).max():,.2f}")
    
    with col4:
        st.metric("Mínimo (30d)", f"${df['Low'].tail(30).min():,.2f}")
    
    # Seção de previsão
    st.subheader("Previsão para Amanhã")
    
    if st.button("Gerar Previsão para Amanhã", type="primary"):
        try:
            with st.spinner("Gerando previsão..."):
                # Aplica engenharia de features
                df_features = feature_engineering.create_features(df)
                
                if df_features.empty:
                    st.error("Dados insuficientes para gerar previsão")
                    return
                
                # Faz previsão
                prediction, confidence = predict.make_prediction(df_features)
                
                # Exibe resultado
                if prediction == 1:
                    st.success(f"### Tendência para amanhã: **ALTA** 📈 (Confiança: {confidence:.2%})")
                    st.info("O modelo prevê que o preço do Bitcoin subirá amanhã.")
                else:
                    st.error(f"### Tendência para amanhã: **QUEDA** 📉 (Confiança: {confidence:.2%})")
                    st.info("O modelo prevê que o preço do Bitcoin cairá amanhã.")
                
        except FileNotFoundError:
            st.error("Modelo não encontrado. Execute o treinamento primeiro.")
        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")

    # Gráfico de preços
    st.subheader("Histórico de Preços do Bitcoin")
    
    fig = px.line(df.reset_index(), x='date', y='Close', 
                  title="Preço de Fechamento do BTC (USD)",
                  labels={'date': 'Data', 'Close': 'Preço (USD)'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Mostra as métricas do modelo
    st.subheader("Métricas do Modelo")
    col1, col2 = st.columns(2)

    try:
        with open("src/ModelHandler/metrics.json", "r") as f:
            metrics = json.load(f)
        with col1:
            st.metric("Acurácia", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}")

        # Matriz de confusão
        with st.expander("Matriz de Confusão"):
            conf_matrix = metrics.get("confusion_matrix")
            if conf_matrix:
                z = conf_matrix
                x = ['Queda', 'Alta']
                y = ['Queda', 'Alta']
                fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
                fig_cm.update_layout(title='Matriz de Confusão')
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.warning("Matriz de confusão não encontrada.")

    except FileNotFoundError:
        st.warning("Métricas não encontradas. Treine o modelo para gerá-las.")
    
    # Informações adicionais
    with st.expander("Informações do Dataset"):
        st.write(f"**Período dos dados:** {df.index.min().strftime('%Y-%m-%d')} até {df.index.max().strftime('%Y-%m-%d')}")
        st.write(f"**Total de registros:** {len(df)}")
        st.write("**Colunas:** Open, High, Low, Close, Volume")
        st.write("**Indicadores do Modelo:** SMA_10, SMA_30, RSI, EMA_12, EMA_26, MACD, MACD_signal, Bollinger_Upper, Bollinger_Lower, Stochastic_K, Stochastic_D, month, day_of_week, day_of_month, close_7_days_ago, close_30_days_ago, volume_change_pct")

if __name__ == "__main__":
    main()