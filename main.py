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
from src.Orchestration.update_scheduler import daily_update_task

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

    # Load data to get the minimum date
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)

    # Date controls for manual update
    st.subheader("Manual Data Update")
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
        if st.button("Update Data Range"):
            logger.info(f"User '{username}' clicked 'Update Data Range'.")
            with st.spinner("Updating data for the selected range..."):
                df_new = data_api.get_btc_data(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                if not df_new.empty:
                    data_handler.update_data(df_new, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    st.success("Data updated for the selected range!")
                    st.rerun() # Rerun to reflect changes
                else:
                    st.error("Error updating data")

    df = data_handler.load_data()
    if df.empty:
        st.warning("No data found. The application is fetching the initial data in the background. Please wait a moment and refresh.")
        return

    config = model_db_handler.load_config(username)
    feature_params = config.get("feature_params", {})

    st.subheader("Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
    with col2:
        change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        changepct = (change / df['Close'].iloc[-2]) * 100
        st.metric("Daily Change", f"${change:,.2f}", f"%{changepct:+.2f}")
    with col3:
        st.metric("30d High", f"${df['High'].tail(30).max():,.2f}")
    with col4:
        st.metric("30d Low", f"${df['Low'].tail(30).min():,.2f}")

    st.subheader("Tomorrow's Forecast")
    if st.button("Generate Tomorrow's Forecast", type="primary"):
        logger.info(f"User '{username}' clicked 'Generate Tomorrow's Forecast'.")
        try:
            with st.spinner("Generating forecast..."):
                model, metrics = model_db_handler.load_model(username)
                if model is None:
                    st.error("Model not found. Please train the model first on the Settings page.")
                else:
                    loaded_feature_params = metrics.get("feature_params", feature_params)
                    df_features = feature_engineering.create_features(df, params=loaded_feature_params)
                    
                    if df_features.empty:
                        st.error("Insufficient data to generate forecast")
                    else:
                        prediction, confidence = predict.make_prediction(df_features, model, metrics)
                        if prediction == 1:
                            st.success(f"### Trend for tomorrow: **UP** 📈 (Confidence: {confidence:.2%})")
                        else:
                            st.error(f"### Trend for tomorrow: **DOWN** 📉 (Confidence: {confidence:.2%})")
        except Exception as e:
            logger.error(f"Error generating forecast for user '{username}': {e}", exc_info=True)
            st.error(f"Error generating forecast: {e}")

    st.subheader("Bitcoin Price History")
    
    period = st.radio("Select period", ["1 month", "3 months", "1 year", "All"], index=3, horizontal=True)

    if period != "All":
        end_date_dt = df.index.max()
        if period == "1 month":
            start_date_dt = end_date_dt - pd.DateOffset(months=1)
        elif period == "3 months":
            start_date_dt = end_date_dt - pd.DateOffset(months=3)
        else: # 1 year
            start_date_dt = end_date_dt - pd.DateOffset(years=1)
        df_filtered = df[df.index >= start_date_dt]
    else:
        df_filtered = df

    fig = px.line(df_filtered.reset_index(), x='date', y='Close', title="BTC Close Price (USD)", labels={'date': 'Date', 'Close': 'Price (USD)'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Dataset Information"):
        st.write(f"**Data period:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        st.write(f"**Total records:** {len(df)}")
        st.write("**Columns:** Open, High, Low, Close, Volume")
        _, metrics = model_db_handler.load_model(username)
        if metrics:
            st.write("**Model Indicators:**", ", ".join(metrics.get("features", [])))

def settings_page(username):
    logger.info(f"Displaying settings page for user '{username}'.")
    st.title("Settings")
    
    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)
    
    config = model_db_handler.load_config(username)
    model_params_saved = config.get("model_params", {})
    feature_params_saved = config.get("feature_params", {})

    with st.expander("Model Training Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.number_input("Number of Estimators", min_value=1, value=model_params_saved.get('n_estimators', 100), key="n_estimators", help="Number of decision trees in the model. More trees can improve accuracy but also increase training time.")
            max_depth = st.number_input("Maximum Depth", min_value=-1, value=model_params_saved.get('max_depth', -1), key="max_depth", help="Maximum depth of each tree. A larger value can capture more complex patterns but may also lead to overfitting. -1 means no limit.")
            reg_alpha = st.number_input("L1 Regularization (Alpha)", min_value=0.0, value=model_params_saved.get('reg_alpha', 0.0), step=0.01, key="reg_alpha", help="L1 regularization term. Helps prevent overfitting by penalizing large weights. Useful when there are many features.")
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=0.01, value=model_params_saved.get('learning_rate', 0.1), step=0.01, key="learning_rate", help="Learning rate. A smaller value makes learning more robust but requires more trees (n_estimators).")
            num_leaves = st.number_input("Number of Leaves", min_value=2, value=model_params_saved.get('num_leaves', 31), key="num_leaves", help="Maximum number of leaves in a tree. A larger value increases model complexity.")
            reg_lambda = st.number_input("L2 Regularization (Lambda)", min_value=0.0, value=model_params_saved.get('reg_lambda', 0.0), step=0.01, key="reg_lambda", help="L2 regularization term. Also helps prevent overfitting, similar to L1.")

    with st.expander("Technical Indicator Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Moving Averages")
            sma_short = st.number_input("Short SMA", min_value=1, value=feature_params_saved.get('sma_window_1', 10), key="sma_short")
            sma_long = st.number_input("Long SMA", min_value=1, value=feature_params_saved.get('sma_window_2', 30), key="sma_long")
            ema_short = st.number_input("Short EMA", min_value=1, value=feature_params_saved.get('ema_window_1', 12), key="ema_short")
            ema_long = st.number_input("Long EMA", min_value=1, value=feature_params_saved.get('ema_window_2', 26), key="ema_long")
        with col2:
            st.subheader("Oscillators")
            rsi_period = st.number_input("RSI Period", min_value=1, value=feature_params_saved.get('rsi_window', 14), key="rsi_period")
            stochastic_window = st.number_input("Stochastic Window", min_value=1, value=feature_params_saved.get('stochastic_window', 14), key="stochastic_window")
            bollinger_window = st.number_input("Bollinger Window", min_value=1, value=feature_params_saved.get('bollinger_window', 20), key="bollinger_window")
        with col3:
            st.subheader("MACD")
            macd_fast = st.number_input("Fast MACD", min_value=1, value=feature_params_saved.get('macd_fast', 12), key="macd_fast")
            macd_slow = st.number_input("Slow MACD", min_value=1, value=feature_params_saved.get('macd_slow', 26), key="macd_slow")
            macd_signal = st.number_input("Signal MACD", min_value=1, value=feature_params_saved.get('macd_signal', 9), key="macd_signal")

    st.subheader("Training Period")
    col1, col2 = st.columns(2)
    with col1:
        train_start_date = st.date_input("Training Start Date", value=min_date)
    with col2:
        train_end_date = st.date_input("Training End Date", value=datetime.now())

    if st.button("Train Model"):
        logger.info(f"User '{username}' clicked 'Train Model'.")
        with st.spinner("Training model..."):
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
                st.success("Model trained successfully!")
            except Exception as e:
                logger.error(f"Error during training for user '{username}': {e}", exc_info=True)
                st.error(f"Error during training: {e}")

    _, metrics = model_db_handler.load_model(username)
    if metrics:
        st.subheader("Current Model Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        with col2:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
        with st.expander("Confusion Matrix"):
            conf_matrix = metrics.get("confusion_matrix")
            if conf_matrix:
                z = conf_matrix
                x = ['Down', 'Up']
                y = ['Down', 'Up']
                fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
                fig_cm.update_layout(title='Confusion Matrix')
                st.plotly_chart(fig_cm, use_container_width=True, key="confusion_matrix_chart")
    else:
        st.warning("Metrics not found. Train the model to generate them.")

def backtesting_page(username):
    logger.info(f"Displaying backtesting page for user '{username}'.")
    st.title("Strategy Backtesting")

    model, metrics = model_db_handler.load_model(username)
    if not model or not metrics:
        st.warning("Model or metrics not found. Train a model first on the Settings page.")
        return

    df_temp = data_handler.load_data()
    min_date = df_temp.index.min().date() if not df_temp.empty else date(2009, 1, 3)

    st.subheader("Backtest Period")
    col1, col2 = st.columns(2)
    with col1:
        backtest_start_date = st.date_input("Backtest Start Date", value=min_date)
    with col2:
        backtest_end_date = st.date_input("Backtest End Date", value=datetime.now())

    if st.button("Start Backtest", type="primary"):
        logger.info(f"User '{username}' clicked 'Start Backtest'.")
        with st.spinner("Running backtest... This may take a few minutes."):
            try:
                feature_cols = metrics.get("features")
                feature_params = metrics.get("feature_params")

                if not feature_cols or not feature_params:
                    st.error("Feature information not found in metrics. Please train the model again.")
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

                st.subheader("Backtest Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Strategy Return", f"{results['total_return_pct']:.2f}%")
                col1.metric("Buy & Hold Return", f"{results['buy_and_hold_return_pct']:.2f}%")
                col2.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                col2.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                col3.metric("Total Trades", results['total_trades'])
                col3.metric("Win Rate", f"{results['win_rate']:.2f}%")

                st.subheader("Portfolio Evolution")
                portfolio_history = results['portfolio_history']['value']
                buy_and_hold_history = results['buy_and_hold_history']
                
                chart_data = pd.concat([portfolio_history, buy_and_hold_history], axis=1)
                chart_data.columns = ['Strategy', 'Buy & Hold']
                
                st.line_chart(chart_data)

                st.subheader("Trade History")
                st.dataframe(trades_history)

            except Exception as e:
                logger.error(f"An error occurred during backtest for user '{username}': {e}", exc_info=True)
                st.error(f"An error occurred during the backtest: {e}")

def login_page():
    logger.info("Displaying login page.")
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        logger.info(f"Login attempt for user '{username}'.")
        token = auth.login_user(username, password)
        if token:
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Don't have an account? Sign up"):
        logger.info("Navigating to registration page.")
        st.session_state['page'] = 'registration'
        st.rerun()

def registration_page():
    logger.info("Displaying registration page.")
    st.title("Registration")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        logger.info(f"Registration attempt for user '{username}'.")
        if auth.create_user(username, password):
            st.success("User created successfully! Please log in.")
            st.session_state['page'] = 'login'
            st.rerun()
        else:
            st.error("User already exists")

    if st.button("Already have an account? Log in"):
        logger.info("Navigating to login page.")
        st.session_state['page'] = 'login'
        st.rerun()

def main():
    if st.session_state.get('authentication_status', False):
        username = st.session_state['username']
        logger.info(f"User '{username}' is logged in.")
        
        st.sidebar.title(f"Welcome, {username}")
        page = st.sidebar.radio("Select a page", ["Dashboard", "Settings", "Backtesting"])

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