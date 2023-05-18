import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go

app = dash.Dash()

def get_graph1():
    df_entidad = pd.read_csv('downloads/Ors_entidad.csv',encoding='cp1252', sep=',', on_bad_lines='warn')
    fig = px.histogram(df_entidad, y='ENTIDAD')
    return fig

def get_graph2():
    df_entidad = pd.read_csv('downloads/Ors_entidad.csv',encoding='cp1252', sep=',', on_bad_lines='warn')
    fig2 = px.histogram(df_entidad, y='RAMO')
    return fig2

def run_time_series_regression():
    df_entidad = pd.read_csv('downloads/Ors_entidad.csv', encoding='cp1252', sep=',', on_bad_lines='warn') #ES AQUÍ: CAMBIAR EL FORMATO DE LAS FECHAS A INT O FLOAT, ESO ESTÁ MODIFICANDO EL RESULTADO, ERROR.
    print(df_entidad.head())
    # Preparar los datos de series de tiempo
    df_entidad['FECHA DE CORTE'] = pd.to_datetime(df_entidad['FECHA DE CORTE'])
    df_entidad.set_index('FECHA DE CORTE', inplace=True)  # Establecer la columna de fecha como el índice del DataFrame
    df_entidad.sort_index(inplace=True)  # Ordenar el DataFrame por fecha ascendente
    
    # Dividir los datos en variables dependientes e independientes
    X = df_entidad[['PRIMA EMITIDA', 'COMISION DIRECTA']]  # Variables independientes
    y = df_entidad['MONTO DE SINIESTRALIDAD']  # Variable dependiente
    
    # Agregar componentes de tendencia y estacionalidad
    X = sm.add_constant(X)  # Agregar una columna constante para el intercepto
    X['TENDENCIA'] = range(len(X))  # Agregar una columna con números enteros consecutivos para representar la tendencia
    
    # Ajustar el modelo de regresión lineal con series de tiempo
    model = sm.OLS(y, X())
    #est = sm.OLS(y, X.astype(float)).fit()
    results = model.fit()
    
    return results

# Llamar a la función y obtener los resultados del modelo
results = run_time_series_regression()

#Prepara los datos de series de tiempo convirtiendo la columna de fecha en el tipo de dato datetime, estableciéndola como el índice del DataFrame y ordenando el DataFrame por fecha ascendente.
#Divide los datos en variables independientes X (en este caso, "Prima emitida" y "Comision Directa") y la variable dependiente y ("Monto de Siniestralidad").
#Agrega componentes de tendencia y estacionalidad al conjunto de variables independientes. En este ejemplo, agregamos una columna constante para el intercepto y una columna con números enteros consecutivos para representar la tendencia.
#Ajusta el modelo de regresión lineal con series de tiempo utilizando sm.OLS() de statsmodels.
#Devuelve los resultados del modelo.
# Crear el gráfico dinámico
fig3 = go.Figure()

# Agregar la serie de tiempo original
fig3.add_trace(go.Scatter(
    x=df_entidad.index,
    y=y,
    mode='lines',
    name='Serie de tiempo original'
))

# Agregar la serie de tiempo predicha
fig3.add_trace(go.Scatter(
    x=X_pred.index,
    y=y_pred,
    mode='lines',
    name='Serie de tiempo predicción'
))

# Personalizar el diseño del gráfico
fig3.update_layout(
    title='Regresión lineal por serie de tiempo',
    xaxis_title='Fecha',
    yaxis_title='Monto de Siniestralidad'
)

# Mostrar el gráfico
fig3.show()

#Utiliza el código anterior para obtener los resultados del modelo de regresión lineal.
#Crea una copia de las variables independientes X y genera 12 puntos adicionales en la columna "Tendencia" para predecir el futuro.
#Utiliza los resultados del modelo para predecir los valores de la serie de tiempo utilizando results.predict().
#Crea un objeto go.Figure() de Plotly para el gráfico dinámico.
#Agrega dos trazas (go.Scatter()) al gráfico: una para la serie de tiempo original y otra para la serie de tiempo predicha.
#Personaliza el diseño del gráfico estableciendo un título, etiquetas de los ejes y otros atributos mediante fig.update_layout().
#Finalmente, puedes llamar a la función plot_time_series_regression() para obtener el gráfico dinámico y mostrarlo en tu aplicación Dash según sea necesario.

app.layout = html.Div([
    html.H1("EVIDENCIA 4", style={"textAlign":'center'}),
    html.H3("Paloma Pardo"),
    html.H3("Omar Spíndola"),
    html.H3("Martín Guzmán"),
    html.H3("Santiago Peña"),
    html.H3("Ricardo Álvarez"),
    html.P("A continuación presentaremos la evidencia 4 del equipo 3 de CCM."),
    html.H3(f"Total de siniestros: {get_total_siniestros()}"),
    html.H3('Resultados de Regresión Lineal con Series de Tiempo Summary:'),
    html.H3(print(results.summary())),
    html.H3('resultados del modelo de regresión lineal Gráfica:'),
    html.H3(print(run_time_series_regression())),
    dcc.Graph(id="graph1", figure = get_graph1()),
    dcc.Graph(id="graph2", figure = get_graph2())])


#1) Construir un dashboard web que integre el análisis de datos y los modelos de aprendizaje construidos, de manera que el socio formador encuentre el valor que estaba buscando para la solución de su problemática. El referido dashboard debe considerar los siguientes gráficos dinámicos (esto es, que cambien/actualicen en función de inputs que el usuario ingrese):
#Pirámide poblacional
#Histórico de prima emitida
#Modalidad de póliza
#Forma de venta
#Distribución por entidad federativa
#Pirámide de siniestros
#Cobertura
#Causa del siniestro (top 15)
#Otros que consideres relevantes

#2) Implantar el/los modelo(s) de regresión lineal con series de tiempo que construiste previamente para predecir valores de "Prima emitida", "Comision Directa" y "Monto de Siniestralidad". Permite al usuario del dashboard puede ingresar, dinámicamente, parametros que sirvan de input a tu(s) modelo(s) y muestra los resultados de manera gráfica/visual; 
#
#3) Presentar de manera apropiada los detalles técnicos contenidos en el referido dashboard.

if __name__ == '__main__':
    app.run_server()

