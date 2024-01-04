'''Fabiano de Lima Navarro

'''
# Instalando as bibliotecas
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
#from fbprophet import Prophet

#  importar os dados.
#url = 'covid_19_data.csv'
#df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update']) # para campos data

URL = 'https://drive.google.com/file/d/1L-WHma4BIWFKJ7X9P897hqtqQU2P85UU/view?usp=drive_link'

df = pd.read_csv('https://drive.google.com/uc?export=download&id='+URL.split('/')[-2], parse_dates=['ObservationDate', 'Last Update']) 

#print (df) # para exibir os dados    // #print (df.dtypes) # conferir os tipos de cada coluna

def corrige_colunas(col_name):    # corrigir nomes das colunas não devem ter letras maiúsculas e nem caracteres especiais
    return re.sub(r"[/| ]", "", col_name).lower()
#print(corrige_colunas("AbcED/P ou")) # testando corrige nome colunas

df.columns = [corrige_colunas(col) for col in df.columns]
#print (df)

# Análises  //  Vamos primeiro selecionar apenas os dados do Brasil para investigar.

#def pais (brasil):
pais = df.loc[df.countryregion == 'Brazil']
print("Brasil", pais)

brasil = df.loc[(df.countryregion == 'Brazil') & (df.confirmed > 0)]
# print ("Confirmados no Brasil", brasil)

#casos confirmados   // Grafico da evolução de casos confirmados

confirmado = px.line(brasil, 'observationdate', 'confirmed', 
        labels={'observationdate':'Data', 'confirmed':'Número de casos confirmados'},
       title='Casos confirmados no Brasil')
confirmado.show()
# novos casos por dia  //  Vamos implementar uma função para fazer a contagem de novos casos
brasil['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x-1],
    np.arange(brasil.shape[0])
))

print("casos Brasil", brasil)

# Visualizando
px_brasil = px.line(brasil, x='observationdate', y='novoscasos', title='Novos casos por dia',
       labels={'observationdate': 'Data', 'novoscasos': 'Novos casos'})
px_brasil.show()

# Calculando as  Mortes no periodo

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes', mode='lines+markers',
              line=dict(color='red'))
)
#Edita o layout
fig.update_layout(title='Mortes por COVID-19 no Brasil',
                   xaxis_title='Data',
                   yaxis_title='Número de mortes')
fig.show()

# Taxa Crescimento // Vamos calcular a taxa de crescimento do COVID desde o primeiro caso.

def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
    # Se data_inicio for None, define como a primeira data disponível no dataset
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)
    
    # Define os valores de presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]
    
    # Define o número de pontos no tempo q vamos avaliar
    n = (data_fim - data_inicio).days
    
    # Calcula a taxa
    taxa = (presente/passado)**(1/n) - 1
    
    return taxa*100

cresc_medio = taxa_crescimento(brasil, 'confirmed')

print(f"O crescimento médio do COVID no Brasil no período avaliado foi de {cresc_medio.round(2)}%.")

# Agora, vamos observar o comportamento da taxa de crescimento no tempo. Para isso, vamos definir uma função para calcular a taxa de crescimento diária.

def taxa_crescimento_diaria(data, variable, data_inicio=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
        
    data_fim = data.observationdate.max()
    n = (data_fim - data_inicio).days
    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
        range(1,n+1)
    ))
    return np.array(taxas)*100
tx_dia = taxa_crescimento_diaria(brasil, 'confirmed')
#print(tx_dia)

primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()
primeiro = px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
        y=tx_dia, title='Taxa de crescimento de casos confirmados no Brasil',
       labels={'y':'Taxa de crescimento', 'x':'Data'})

cresc_dia = taxa_crescimento_diaria(brasil, 'confirmed')

primeiro.show()

#Predições
#Vamos construir um modelo de séries temporais para prever os novos casos. Antes analisemos a série temporal.


novoscasos = brasil.novoscasos
novoscasos.index = brasil.observationdate

res = seasonal_decompose(novoscasos)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.scatter(novoscasos.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='red')
plt.show()

#Decompondo a série de confirmados
confirmados = brasil.confirmed
confirmados.index = brasil.observationdate
res2 = seasonal_decompose(confirmados)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
title="série de confirmados no Brasil"
ax1.plot(res2.observed)
ax2.plot(res2.trend)
ax3.plot(res2.seasonal)
ax4.scatter(confirmados.index, res2.resid)
ax4.axhline(0, linestyle='dashed', c='red')
plt.show()

# Predizendo o número de casos confirmados com um AUTO-ARIMA  // !pip install pmdarima
# from pmdarima.arima import auto_arima
modelo = auto_arima(confirmados)
pd.date_range('2020-05-01', '2020-05-19')
fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observed'
))

fig.add_trace(go.Scatter(x=confirmados.index, y = modelo.predict_in_sample(), name='Predicted'))

fig.add_trace(go.Scatter(x=pd.date_range('2020-05-20', '2020-06-05'), y=modelo.predict(15), name='Forecast'))

fig.update_layout(title='Previsão de casos confirmados para os próximos 15 dias',
                 yaxis_title='Casos confirmados', xaxis_title='Data')
fig.show()
'''
###########################################
# Modelo de crescimento //iremos usar fbprophet  /// !conda install -c conda-forge fbprophet -y

# from fbprophet import Prophet
!conda install -c conda-forge fbprophet -y
# preparando os dados
train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

# renomeia colunas
train.rename(columns={"observationdate":"ds","confirmed":"y"},inplace=True)
test.rename(columns={"observationdate":"ds","confirmed":"y"},inplace=True)
test = test.set_index("ds")
test = test['y']

profeta = Prophet(growth="logistic", changepoints=['2020-03-21', '2020-03-30', '2020-04-25', '2020-05-03', '2020-05-10'])

#pop = 1000000
pop = 211463256 #https://www.ibge.gov.br/apps/populacao/projecao/box_popclock.php
train['cap'] = pop

# Treina o modelo
profeta.fit(train)

# Construindo previsões para o futuro
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop
forecast =  profeta.predict(future_dates)

fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
fig.add_trace(go.Scatter(x=test.index, y=test, name='Observados - Teste'))
fig.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
fig.update_layout(title='Predições de casos confirmados no Brasil')
fig.show()

'''




















