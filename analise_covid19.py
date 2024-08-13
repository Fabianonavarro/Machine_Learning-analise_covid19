import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from prophet import Prophet
import warnings
import streamlit as st

# Suprimir warnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

# URL padrão dos dados
URL = 'https://drive.google.com/uc?export=download&id=1L-WHma4BIWFKJ7X9P897hqtqQU2P85UU'

def carrega_dados(url):
    """Carrega dados de COVID-19 a partir de uma URL e corrige os nomes das colunas."""
    try:
        df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])
        df.columns = [col.lower().replace(" ", "_").replace("/", "").replace("|", "") for col in df.columns]
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro
    return df

def filtra_dados_brasil(df):
    """Filtra os dados para o Brasil e remove entradas com casos confirmados iguais a zero."""
    if df.empty:
        st.warning("O DataFrame está vazio. Verifique a carga dos dados.")
        return pd.DataFrame()
    brasil = df.loc[(df.countryregion == 'Brazil') & (df.confirmed > 0)]
    return brasil.copy()

def mostrar_tabela(df):
    """Mostra a tabela com os dados resumidos."""
    if df.empty:
        st.warning("Os dados estão vazios. Não é possível mostrar a tabela.")
        return
    st.subheader("Tabela de Dados")
    st.write(df)

def plota_confirmados(brasil):
    """Plota a evolução dos casos confirmados no Brasil."""
    if brasil.empty:
        st.warning("Os dados de Brasil estão vazios. Não é possível plotar.")
        return
    fig = px.line(brasil, x='observationdate', y='confirmed', 
                  labels={'observationdate': 'Data', 'confirmed': 'Número de casos confirmados'},
                  title='Casos confirmados no Brasil')
    st.plotly_chart(fig)

def calcula_novos_casos(brasil):
    """Calcula e plota o número de novos casos por dia."""
    if brasil.empty:
        st.warning("Os dados de Brasil estão vazios. Não é possível calcular novos casos.")
        return
    brasil['novoscasos'] = brasil['confirmed'].diff().fillna(0)
    fig = px.line(brasil, x='observationdate', y='novoscasos', 
                  title='Novos casos por dia', 
                  labels={'observationdate': 'Data', 'novoscasos': 'Novos casos'})
    st.plotly_chart(fig)

def plota_mortes(brasil):
    """Plota o número de mortes por COVID-19 no Brasil."""
    if brasil.empty:
        st.warning("Os dados de Brasil estão vazios. Não é possível plotar mortes.")
        return
    if 'deaths' not in brasil.columns:
        st.warning("Coluna 'deaths' não encontrada nos dados.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes', mode='lines+markers',
                             line=dict(color='red')))
    fig.update_layout(title='Mortes por COVID-19 no Brasil', xaxis_title='Data', yaxis_title='Número de mortes')
    st.plotly_chart(fig)

def decompoe_serie(novoscasos, confirmados):
    """Decompõe e plota as séries temporais de novos casos e casos confirmados."""
    if novoscasos.empty or confirmados.empty:
        st.warning("Os dados estão vazios. Não é possível decompor a série.")
        return

    if not isinstance(novoscasos.index, pd.DatetimeIndex):
        novoscasos.index = pd.to_datetime(novoscasos.index)
    if not isinstance(confirmados.index, pd.DatetimeIndex):
        confirmados.index = pd.to_datetime(confirmados.index)

    try:
        res_novoscasos = seasonal_decompose(novoscasos, period=7)
        res_confirmados = seasonal_decompose(confirmados, period=7)
    except Exception as e:
        st.error(f"Erro na decomposição da série: {e}")
        return

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(res_novoscasos.observed)
    axs[0].set_title('Novos Casos - Observado')
    axs[1].plot(res_novoscasos.trend)
    axs[1].set_title('Novos Casos - Tendência')
    axs[2].plot(res_novoscasos.seasonal)
    axs[2].set_title('Novos Casos - Sazonal')
    axs[3].scatter(novoscasos.index, res_novoscasos.resid)
    axs[3].axhline(0, linestyle='dashed', c='red')
    axs[3].set_title('Novos Casos - Resíduo')
    plt.tight_layout()
    st.pyplot(fig)

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(res_confirmados.observed)
    axs[0].set_title('Casos Confirmados - Observado')
    axs[1].plot(res_confirmados.trend)
    axs[1].set_title('Casos Confirmados - Tendência')
    axs[2].plot(res_confirmados.seasonal)
    axs[2].set_title('Casos Confirmados - Sazonal')
    axs[3].scatter(confirmados.index, res_confirmados.resid)
    axs[3].axhline(0, linestyle='dashed', c='red')
    axs[3].set_title('Casos Confirmados - Resíduo')
    plt.tight_layout()
    st.pyplot(fig)

def previsao_arima(confirmados):
    """Cria e plota previsões usando o modelo ARIMA."""
    if confirmados.empty:
        st.warning("Dados para ARIMA estão vazios. Verifique os dados de entrada.")
        return

    if not isinstance(confirmados.index, pd.DatetimeIndex):
        confirmados.index = pd.to_datetime(confirmados.index)

    modelo = auto_arima(confirmados, seasonal=True, m=7, trace=True, error_action='ignore', suppress_warnings=True)

    previsao_in_sample = modelo.predict_in_sample()

    forecast_periods = 15
    future_dates = pd.date_range(start=confirmados.index[-1] + pd.Timedelta(days=1), periods=forecast_periods)
    forecast = modelo.predict(n_periods=forecast_periods)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=confirmados.index, y=confirmados, name='Observado'))
    fig.add_trace(go.Scatter(x=confirmados.index, y=previsao_in_sample, name='Predito', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, name='Previsão', line=dict(dash='dash')))
    fig.update_layout(title='Previsão ARIMA dos Casos Confirmados', xaxis_title='Data', yaxis_title='Número de casos confirmados')
    st.plotly_chart(fig)

def previsao_prophet(brasil):
    """Cria e plota previsões usando o modelo Prophet."""
    if brasil.empty:
        st.warning("Dados para Prophet estão vazios. Verifique os dados de entrada.")
        return

    if 'observationdate' not in brasil.columns or 'confirmed' not in brasil.columns:
        st.warning("Colunas necessárias para Prophet não encontradas.")
        return

    df_prophet = brasil[['observationdate', 'confirmed']].rename(columns={'observationdate': 'ds', 'confirmed': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Observado'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Limite Inferior', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Limite Superior', line=dict(dash='dash')))
    fig.update_layout(title='Previsão Prophet dos Casos Confirmados', xaxis_title='Data', yaxis_title='Número de casos confirmados')
    st.plotly_chart(fig)

def main():
    """Função principal para executar o aplicativo Streamlit."""
    st.title("Análise de Dados de COVID-19")

    # Carregar dados
    df = carrega_dados(URL)
    brasil = filtra_dados_brasil(df)
    
    menu = st.sidebar.radio("Escolha uma opção", ("Visualização de Dados", "Análise de Séries Temporais"))

    if menu == "Visualização de Dados":
        st.sidebar.subheader("Visualização de Dados")
        visualizacao = st.sidebar.radio("Escolha a visualização", ["Tabela", "Casos Confirmados", "Novos Casos", "Mortes"])

        if visualizacao == "Tabela":
            mostrar_tabela(brasil)
        elif visualizacao == "Casos Confirmados":
            plota_confirmados(brasil)
        elif visualizacao == "Novos Casos":
            calcula_novos_casos(brasil)
        elif visualizacao == "Mortes":
            plota_mortes(brasil)

    elif menu == "Análise de Séries Temporais":
        st.sidebar.subheader("Análise de Séries Temporais")
        analise = st.sidebar.radio("Escolha a análise", ["Decomposição", "Previsão ARIMA", "Previsão Prophet"])

        if analise == "Decomposição":
            if 'novoscasos' not in brasil.columns:
                brasil['novoscasos'] = brasil['confirmed'].diff().fillna(0)
            decompoe_serie(brasil['novoscasos'], brasil['confirmed'])
        elif analise == "Previsão ARIMA":
            previsao_arima(brasil['confirmed'])
        elif analise == "Previsão Prophet":
            previsao_prophet(brasil)

if __name__ == "__main__":
    main()
