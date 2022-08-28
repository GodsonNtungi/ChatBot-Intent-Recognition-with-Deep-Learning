import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer  ## reduce a word to its  stem worked >> work
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intense.json').read())
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents["intents"]:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_words]
words = sorted(set(words))
print(classes)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []

for document in documents:
    bag = []
    output = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    for value in classes:
        output.append(1) if value == document[1] else output.append(0)

    training.append([bag, output])
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
print(train_x)
print(train_y)

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(len(train_x[0]),)))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='sigmoid'))

sgd = SGD(lr =0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5)
model.save('chatbot.h5')
print('super')
