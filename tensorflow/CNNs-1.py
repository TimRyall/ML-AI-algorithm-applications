import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import datetime

####################################### import data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalise

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


################################### shuffle data
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

############################## CNN
model =  tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32,3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
          ])

################### add optimiser and loss
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',metrics=['accuracy'])

######################## fit the model
model.fit(train_ds, 
          epochs=5, 
          validation_data=test_ds)
model.summary()


