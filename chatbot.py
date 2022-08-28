import json
import pickle
import random
from tensorflow import keras
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer  ## reduce a word to its  stem worked >> work
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import warnings
warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intense.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


class Chatbot:
    def train(self):
        words_train = []
        classes_train = []
        documents = []
        ignore_words = ['?', '!', '.', ',']

        for intent in intents["intents"]:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                words_train.extend(word_list)
                documents.append((word_list, intent['tag']))
                if intent['tag'] not in classes_train:
                    classes_train.append(intent['tag'])
        words_train = [lemmatizer.lemmatize(word) for word in words_train if word not in ignore_words]
        words_train = sorted(set(words_train))
        print(classes_train)

        pickle.dump(words_train, open('words.pkl', 'wb'))
        pickle.dump(classes_train, open('classes.pkl', 'wb'))

        training = []

        for document in documents:
            bag = []
            output = []
            word_patterns = document[0]
            word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

            for word in words_train:
                bag.append(1) if word in word_patterns else bag.append(0)

            for value in classes_train:
                output.append(1) if value == document[1] else output.append(0)

            training.append([bag, output])
        random.shuffle(training)
        training = np.array(training)
        print(training)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        print(train_x)
        print(train_y)

        model = Sequential()

        model.add(Dense(64, activation='relu', input_shape=(len(train_x[0]),)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)
        model.save('chatbot.h5')
        print('super')

    def create_words(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def create_bag(self, sentence):
        bag = []
        sentence_words = self.create_words(sentence)
        for word in words:
            bag.append(1) if word in sentence_words else bag.append(0)

        return np.array(bag)

    def predict(self, sentence):
        model = load_model('chatbot.h5')
        x_predict = self.create_bag(sentence)
        y_predict = model.predict(np.array([x_predict]),verbose=0)
        error_threshold = 0.25
        results = [[i, r] for i, r in enumerate(y_predict[0]) if r > error_threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        # what it does the sort function
        # [[0, 0.40031675], [1, 0.6715881]], results
        # [[1, 0.6715881], [0, 0.40031675]], results

        return results[0][0]

    def generate_output(self, sentence):
        position = self.predict(sentence)
        class_predicted = classes[position]
        result = 'unknown'
        for intent in intents['intents']:
            if intent['tag'] == class_predicted:
                result = random.choice(intent['responses'])
                print(result)

        return result


if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.train()
    while True:
        chatbot.generate_output(input('Put a sentence \n'))
