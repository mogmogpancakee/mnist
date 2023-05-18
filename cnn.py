import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()
model.add(layers.Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

model.save('mnist_model.h5')

model = load_model('mnist_model.h5')

class DigitClassifier(tk.Tk):
    def __init__(self, model, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.canvas_size = 400
        self.image = Image.new('L', (28, 28), (0))
        self.draw = ImageDraw.Draw(self.image)

        self.model = model

        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.bind('<B1-Motion>', self.draw_digit)

        self.button = tk.Button(self, text='Predict', command=self.predict_digit)
        self.button.pack()

    def draw_digit(self, event):
        x = event.x * 28 // self.canvas_size
        y = event.y * 28 // self.canvas_size
        self.draw.ellipse([x, y, x + 1, y + 1], fill=(255))
        self.canvas.create_oval(event.x, event.y, event.x + 1, event.y + 1, fill='black')

    def predict_digit(self):
        digit = np.array(self.image).reshape(1, 28, 28)
        prediction = self.model.predict(digit)[0]
        print('Prediction:', np.argmax(prediction))

        # Clear the image for the next prediction
        self.image = Image.new('L', (28, 28), (0))
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete('all')

app = DigitClassifier(model)
app.mainloop()
