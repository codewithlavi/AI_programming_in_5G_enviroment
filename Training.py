import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD

# Inicializamos el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar archivos necesarios
with open('intents.json') as file:
    intents = json.load(file)

# Cargar palabras y clases si existen, o inicializarlos
try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
except FileNotFoundError:
    words = []
    classes = []

# Procesamos los patrones y etiquetas
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenización
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizamos y eliminamos duplicados
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guardamos las palabras y clases procesadas
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparamos los datos para el entrenamiento
training = []
output_empty = [0] * len(classes)
for document in documents:
    # Bolsa de palabras
    bag = [1 if lemmatizer.lemmatize(word.lower()) in document[0] else 0 for word in words]

    # Etiquetas
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Barajamos los datos de entrenamiento y los convertimos a arrays de numpy
random.shuffle(training)
training = np.array(training, dtype=object)

# Separamos las características (X) y las etiquetas (Y)
train_x = np.array(list(training[:, 0].tolist()))
train_y = np.array(list(training[:, 1].tolist()))

# Creamos el modelo de la red neuronal
model = Sequential()
model.add(Input(shape=(len(train_x[0]),)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configuramos el optimizador y compilamos el modelo
sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenamos el modelo
model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)

# Guardamos el modelo entrenado en formato .keras
model.save("chatbot_model.keras")
