import tensorflow as tf
import pandas as pd

# Loading the MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
df = pd.DataFrame(data=y_test)

#data preprocessing
#normalize x_train and y_train to scale down between 0 and 1 because their values are between 0 and 255
#we do not normalize y_train and y_test because these are the labels 0,1,2,3,4,5,6,7,8,9
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.relu, ))  # activity_regularizer=tf.keras.regularizers.L2(l=0.9)
    # model.add(tf.keras.layers.Dense(units=397, activation=tf.nn.relu, ))#activity_regularizer=tf.keras.regularizers.L2(l=0.9)
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
    # Compiling and optimizing model
    #create loss
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()

    #create optimizer with different learning rates and momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=[scce,mse,'accuracy'])

    return model

def train(models):
    losses = []
    # Training the model
    for i in range(len(models)):
        history = models[i].fit(x_train, y_train, validation_data=(x_test,y_test), epochs=1)
        losses.append(history.history['loss'])
    return  models,losses
