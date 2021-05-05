import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import numpy as np

num_inputs = 2
num_outputs = 2

# Create one layer model
model = tf.keras.Sequential()

# No bias, no activation, initialize weights with 1.0
layer1 = Dense(units=num_inputs, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.ones())
layer2 = Dense(units=num_outputs, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.ones())
model.add(layer1)
model.add(layer2)

# use MSE as loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Arbitrary model input
x = np.array([2.0, 0.5])

# Expected output
y_true = np.array([2.0, 1.0])

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

    # loss gradient with respect to loss input y
    dy_dw = tape.gradient(y, model.trainable_variables)

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
print("dloss_dy={}".format(*[v.numpy() for v in dloss_dy]))

# print weight gradients d_loss/d_w
print("grad=\n{}".format(*[v.numpy() for v in grad]))

# print updated dense layer weights
print("updated weights=")
print(*[v.numpy() for v in model.trainable_variables], sep="\n")

