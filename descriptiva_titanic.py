import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
#import kagglehub

#path = kagglehub.dataset_download("brendan45774/test-file")
#file_path = f"{path}/tested.csv"  # Ajusta "test.csv" si el archivo tiene otro nombre
#titanic = pd.read_csv(file_path)
import seaborn as sns
titanic=sns.load_dataset('titanic')

# Título de la app
st.title("Descriptiva sobre los pasajeros del Titanic")

# Texto introductorio
st.write("¡Navega en los datos!")

# Descripción inicial
st.write("""
### ¡Bienvenidos!
Esta aplicación interactiva permite explorar el dataset de Titanic.
Puedes:
1. Ver los primeros registros.
2. Consultar información general del dataset.
3. Generar gráficos dinámicos.
""")

# Sección para explorar el dataset
st.sidebar.header("Exploración de datos")

# Mostrar las primeras filas dinámicamente
if st.sidebar.checkbox("Mostrar primeras filas"):
    n_rows = st.sidebar.slider("Número de filas a mostrar:", 1, 50, 5)
    st.write(f"### Primeras {n_rows} filas del dataset")
    st.write(titanic.head(n_rows))


# Mostrar información del dataset
import io

if st.sidebar.checkbox("Mostrar información del dataset"):
    columnas_con_nulos = titanic.columns[titanic.isnull().any()].tolist()
    porcentajes_nulos = titanic.isnull().mean() * 100
    st.write("""
    ### Información para las variables numéricas es:
    """)
    st.write(titanic.describe())
    columns_object = titanic.select_dtypes(include=['object']).columns
    frequencies = {}
    for col in columns_object:
      frequencies[col] = titanic[col].value_counts()
    frequencies_df = pd.DataFrame.from_dict(frequencies, orient='index').transpose()
    st.write("### Tablas de Frecuencias para Variables de Tipo categórico en el DataFrame Titanic")
    st.write(frequencies_df)
    st.write("""
    ### Las variables que tienen nulos son:
    """)
    for col in columnas_con_nulos:
      st.write(f"- {col}: {porcentajes_nulos[col]:.2f}%")
    st.write("""
    ### Como se observa, la cantidad de nulos en algunas variables es muy baja y en otras variables es muy alta.
    Por ejemplo en las variables embarked y embark_town el porcentaje de nulos es inferior al 1%, para esos registros
    es factible eliminarlos sin ningún problema. La variable deck tiene cerca de un 77.22% de nulos, lo cual es bastante,
    en este caso lo mejor que puede uno hacer es eliminar la columna, para la
    variable age también es factible usar modelos predictivos para predecir los valores faltantes.
    """)

# Descripción inicial

# Estadísticas descriptivas
if st.sidebar.checkbox("Mostrar estadísticas descriptivas"):
    st.sidebar.header("Gráficos dinámicos")
    st.write("### Estadísticas descriptivas")
    numeric_columns = titanic.select_dtypes(include=[np.number]).columns
    chart_type = st.sidebar.radio(
    "Selecciona el tipo de gráfico:",
    ("Dispersión", "Histograma", "Boxplot", "Lineplot", "Violinplot", "Corrplot"))
    st.write("### Gráficos")
    if chart_type == "Dispersión":
        x_var_num = st.sidebar.selectbox("Selecciona la variable X:", numeric_columns)
        y_var_num = st.sidebar.selectbox("Selecciona la variable Y:", numeric_columns)
        st.write(f"#### Gráfico de dispersión: {x_var_num} vs {y_var_num}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=titanic, x=x_var_num, y=y_var_num, ax=ax)
        st.pyplot(fig)
        st.write("""
        ### Interpretaciones:
        Age vs Fare: A pesar de que no se observa una relación clara entre ambas variables, sin embargo, puede
        observarse que las tarifas más altas, de primera clase, fueron pagadas solamente por personas mayores de edad.

        Age vs SibSp: Se puede observar que los niños pequeños solían viajar más con sus hermanos y las personas mayores
        no.

        Age vs Parch: Se puede observar que los niños pequeños solían viajar más con sus padres/hijos y las personas mayores
        con uno o ninguno de sus padres/hijos.

        Fare vs SibSp: La varianza de los datos es mayor en las tarifas pagadas cuando el número de hermanos/esposos
        es menor.
        """)
    elif chart_type == "Histograma":
        x_var = st.sidebar.selectbox("Selecciona la variable X:", titanic.columns)
        st.write(f"#### Histograma de {x_var}")
        fig, ax = plt.subplots()
        sns.histplot(titanic[x_var], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
        st.write("""
        ### Interpretaciones:
        Edad: La mayoría de los pasajeros tenía entre 20 y 40 años.

        Fare: La mayor cantidad de las tarifas eran bajas y las de primera clase son muy pocas y se evidencian en la cola del histograma.

        Sibsp: La mayoría de los pasajeros viajaban solos.

        Parch: La mayoría de los pasajeros viajaban solos.

        Pclass: La mayoría de los pasajeros viajaban en tercera clase. Seguido por la primera clase.
        """)
    elif chart_type == "Boxplot":
        x_var_num = st.sidebar.selectbox("Selecciona la variable X:", numeric_columns)
        st.write(f"#### Boxplot de {x_var_num}")
        fig, ax = plt.subplots()
        sns.boxplot(data=titanic, x=x_var_num, ax=ax)
        st.pyplot(fig)
        st.write("""
        ### Interpretaciones:
        Edad: La mediana de edad está en torno a los 28 años y el IQR está entre 20 y 38 años

        Fare: La mayor cantidad de las tarifas eran bajas y las de primera clase son muy pocas y se evidencian en la cola del histograma.

        Sibsp: La mayoría de los pasajeros viajaban solos.

        Parch: La mayoría de los pasajeros viajaban solos.

        Pclass: La mayoría de los pasajeros viajaban en tercera clase. Seguido por la primera clase.
        """)
    elif chart_type == "Lineplot":
        x_var = st.sidebar.selectbox("Selecciona la variable X:", titanic.columns)
        y_var = st.sidebar.selectbox("Selecciona la variable Y:", titanic.columns)
        st.write(f"#### Lineplot de {y_var} por {x_var}")
        fig, ax = plt.subplots()
        sns.lineplot(data=titanic, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)
        st.write("""
        ### Interpretaciones:
        Age vs Fare: No hay una correlación evidente entre la edad y la tarifa pagada, ya que los picos y valles
         de ambas variables no coinciden de manera consistente.

        SibSp vs Parch: En general, la variabilidad de ambas líneas es baja, con pocos pasajeros viajando con más de 1 o 2 familiares.

        Pclass vs Fare: Hay una relación visible donde los picos de tarifas altas corresponden a pclass 1,
        mientras que las tarifas bajas corresponden a pclass 3.
        """)

    elif chart_type == "Violinplot":
        x_var = st.sidebar.selectbox("Selecciona la variable X:", titanic.columns)
        y_var = st.sidebar.selectbox("Selecciona la variable Y:", titanic.columns)
        st.write(f"#### Violinplot de {y_var} por {x_var}")
        fig, ax = plt.subplots()
        sns.violinplot(data=titanic, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Corrplot":
        st.write("#### Mapa de Calor de Correlación (Corrplot)")
        corr = titanic[numeric_columns].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.write("""
        ### Edad (Age):

        No hay como tal una correlación alta con las demás variables

        ### Tarifa (Fare):

        Correlación con SibSp: La correlación  es moderada, indicando que aquellos con más familiares a bordo pueden haber pagado
        tarifas más altas.

        Correlación con Parch: Hay una correlación moderada lo que sugiere que los que viajaron con más familiares
        pagaron tarifas más altas.

        Correlación con Pclass: Hay una fuerte correlación negativa entre la tarifa y la clase del
        boleto.

        ### Número de Hermanos/Esposos:

        Correlación con Parch: Existe una correlación positiva entre el número de
        hermanos/esposos y el número de padres/hijos, lo que indica que aquellos que viajaban con un tipo de
        familiar también tendían a viajar con otro tipo de familiar.

        ### Número de Padres/Hijos (Parch):

        Correlación con Pclass: Hay una baja correlación negativa entre el número de
        padres/hijos y la clase del boleto.
        """)

st.write("¡Explora más opciones en la barra lateral!")
