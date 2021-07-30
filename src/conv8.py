import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import numpy as np

input_height = 10
input_width = 10
kernel_size = 5
channels_in = 1
channels_out = 1
strides = 1
# Create sequential model
model = tf.keras.Sequential()

kernel_weights_num = kernel_size * kernel_size * channels_in * channels_out
weights = np.reshape(np.linspace(start=1, stop=kernel_weights_num, num=kernel_weights_num), (kernel_size, kernel_size, channels_in, channels_out))

biases = np.ones(shape=(channels_out,))

model.add(tf.keras.layers.Conv2D(filters=channels_out,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 activation=None,
                                 data_format="channels_last",
                                 padding="same",
                                 name="conv1",
                                 weights=[weights, biases]))

# use MSE as loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Arbitrary model input
x = np.ones(shape=(input_height, input_width, 1))

# Expected output
y_true = np.ones(shape=(1, input_height, input_width, channels_out))

# SGD update rule for parameter w with gradient g when momentum is 0:
# w = w - learning_rate * g
# For simplicity make learning_rate=1.0
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.0)

# Get model output y for input x, compute loss, and record gradients
with tf.GradientTape(persistent=True) as tape:

    # get model output y for input x
    # add newaxis for batch size of 1
    xt = tf.convert_to_tensor(x[np.newaxis, ...])
    tape.watch(xt)
    y = model(xt)

    # obtain MSE loss
    loss = loss_fn(y_true, y)

print("initial weights:")
print(f"conv weights=\n {np.transpose(model.trainable_variables[0].numpy(), [2, 3, 0, 1])}")
print(f"conv bias=\n {model.trainable_variables[1].numpy()}")

# adjust Dense layer weights
grad = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grad, model.trainable_variables))

# print model input and output excluding batch dimention
print(f"input x=\n{np.squeeze(x)}")
print(f"output y=\n{np.transpose(y, [0, 3, 1, 2])}")
print(f"expected output y_true=\n{np.transpose(y_true, [0, 3, 1, 2])}")

# print MSE loss
print(f"loss={loss}")

# print updated dense layer weights
print("updated weights=")
print(f"conv weights=\n {np.transpose(model.trainable_variables[0].numpy(), [2, 3, 0, 1])}")
print(f"conv bias=\n {model.trainable_variables[1].numpy()}")


