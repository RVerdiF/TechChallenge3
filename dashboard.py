import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import src.ApiHandler.data_api as data_api
import src.DataHandler.data_handler as data_handler
import src.DataHandler.feature_engineering as feature_engineering
import src.ModelHandler.predict as predict

st.set_page_config(page_title="Dashboard BTC", layout="wide")

def main():
    st.title("Dashboard de Previsão de Preço do BTC")
    
    # Sidebar para controles
    st.sidebar.header("Controles")
    
    # Botão para atualizar dados
    if st.sidebar.button("Atualizar Dados"):
        with st.spinner("Atualizando dados..."):
            # Busca dados dos últimos 365 dias
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            df = data_api.get_btc_data(start_date=start_date, end_date=end_date)
            if not df.empty:
                data_handler.save_data(df)
                st.sidebar.success("Dados atualizados!")
            else:
                st.sidebar.error("Erro ao atualizar dados")
    
    # Carrega dados existentes
    df = data_handler.load_data()
    
    if df.empty:
        st.warning("Nenhum dado encontrado. Clique em 'Atualizar Dados' para começar.")
        return
    
    # Gráfico de preços
    st.subheader("Histórico de Preços do Bitcoin")
    
    fig = px.line(df.reset_index(), x='date', y='Close', 
                  title="Preço de Fechamento do BTC (USD)",
                  labels={'date': 'Data', 'Close': 'Preço (USD)'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
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
                prediction = predict.make_prediction(df_features)
                
                # Exibe resultado
                if prediction == 1:
                    st.success("### Tendência para amanhã: **ALTA** 📈")
                    st.info("O modelo prevê que o preço do Bitcoin subirá amanhã.")
                else:
                    st.error("### Tendência para amanhã: **QUEDA** 📉")
                    st.info("O modelo prevê que o preço do Bitcoin cairá amanhã.")
                
        except FileNotFoundError:
            st.error("Modelo não encontrado. Execute o treinamento primeiro.")
        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")
    
    # Informações adicionais
    with st.expander("Informações do Dataset"):
        st.write(f"**Período dos dados:** {df.index.min().strftime('%Y-%m-%d')} até {df.index.max().strftime('%Y-%m-%d')}")
        st.write(f"**Total de registros:** {len(df)}")
        st.write("**Colunas:** Open, High, Low, Close, Volume")

if __name__ == "__main__":
    main()