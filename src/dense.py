import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np

num_inputs = 3
num_outputs = 2

# Create one layer model
model = tf.keras.Sequential()

# No bias, no activation, initialize weights with 1.0
model.add(Dense(units=num_outputs, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.ones()))

# use MSE as loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Arbitrary model input
x = np.array([2.0, 0.5, 1])

# Expected output
y_true = np.array([1.5, 1.0])


# SGD update rule for parameter w with gradient g when momentum is 0 is as follows:
#   w = w - learning_rate * g
#
#   For simplicity make learning_rate=1.0
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

    # loss gradient with respect to loss input y
    dloss_dy = tape.gradient(loss, y)

    # adjust Dense layer weights
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

# print model input and output excluding batch dimention
print(f"input x={x}")
print(f"output y={y[0]}")
print(f"expected output y_true={y_true}")

# print MSE loss
print(f"loss={loss}")

# print loss gradients
print(f"dloss_dy={dloss_dy[0].numpy()}")

# print updated dense layer weights
print(f"dense layer weights=\n{model.trainable_variables[0].numpy()}")

