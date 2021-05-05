# Tkct
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

def pproc_num(num):
    binary = bin(num)[2:].zfill(4*6)
    result = []
    for i in range(6):
        idx_begin = 4*i
        idx_end = idx_begin+4
        result.append([int(j) for j in binary[idx_begin:idx_end]])
    return np.array(result)

def load_data():
    numbers = []
    labels = []
    for i in range(167772):
        rnd = random.randrange(16777216)
        pproced_num = pproc_num(rnd)
        numbers.append(pproced_num)
        labels.append(rnd % 2)
    a = np.array
    return (a(numbers[:-100000]), a(labels[:-100000])), (a(numbers[-100000:-(100000-6000)]), a(labels[-100000:-(100000-6000)]))


(train_data, train_labels), (test_data, test_labels) = load_data()

print("shape:")
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=6)

test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


class_names = ['even', 'odd']
predictions = model.predict(np.array(
    [ pproc_num(i) for i in range(1000) ]
))
for i in range(1000):
    result = np.argmax(predictions[i])
    result = class_names[result]
    print(i, result)
