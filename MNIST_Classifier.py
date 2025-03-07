import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import sklearn

def preprocess_image(image):
  image = image.convert('L') # convertir a escala de grises
  image = image.resize((28,28))
  image_array = img_to_array(image) / 255.0
  image_array = np.expand_dims(image_array, axis=0)
  return image_array

def load_model():
  filename = "model_trained_classifier.pkl.gz"
  with gzip.open(filename, 'rb') as f:
    model = pickle.load(f)
  return model

def main():
  st.title("Clasificación de la base de datos MNIST")
  st.markdown("Sube una imagen para clasificar")

  uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG:)", type = ["jpg", "png", "jpeg"])

  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "imagen subida")

    preprocessed_image = preprocess_image(image) # (1,28,28)
   
    st.image(preprocessed_image, caption = "imagen subida")

    if st.button("Clasificar imagen"):
      st.markdown("Imagen clasificada")
      model = load_model()

      prediction = model.predict(preprocessed_image.reshape(1,-1)) # (1, 784)
      st.markdown(f"La imagen fue clasificada como: {prediction}")
      
st.write("El mejor modelo fue un KNeighborsClassifier, este se comparó contra un modelo de DecisionTreeClassifier y resultó siendo el mejor usando el método de GridSearch.")
st.write("""
                Este clasificador tiene los siguientes hiperparámetros:
                - **n_neighbors=4**: Este hiperparámetro determina la cantidad de vecinos cercanos a tomar en cuenta para la predicción. 
                  El modelo como tal considera los 4 puntos de datos más próximos en el espacio de características.
                - **p=3**: Este hiperparámetro se usa para definir la distancia usada para los puntos de datos. 
                  El método de GridSearch eligió la distancia de Minkowski como la mejor distancia, por encima de la distancia euclídea y la de Manhattan.
            """)

if __name__ == "__main__":
  main()
