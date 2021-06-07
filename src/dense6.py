import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import numpy as np

num_inputs = 2
num_outputs = 2

# Create sequential model
model = tf.keras.Sequential()

layer1 = Dense(units=num_inputs, use_bias=True, activation="sigmoid", weights=[np.ones([num_inputs, num_inputs]), np.full([num_inputs], 2.0)])
layer2 = Dense(units=num_outputs, use_bias=True, activation=None, weights=[np.array([[1.0, 2.0], [3.0, 2.0]]), np.array([1.0, 2.0])])
layer3 = Softmax(axis=-1)
model.add(layer1)
model.add(layer2)
model.add(layer3)

# use CCE as loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Arbitrary model input
x = np.array([-1.0, 1.0])

# Expected output
y_true = np.array([[1.0, 0.0]])

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

# adjust Dense layer weights
grad = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grad, model.trainable_variables))

# print model input and output excluding batch dimention
print(f"input x={x}")
print(f"output y={y[0]}")
print(f"expected output y_true={y_true}")

# print MSE loss
print(f"loss={loss}")

# print updated dense layer weights
print("updated weights=")
for idx, layer_vars in enumerate(model.trainable_variables):
    print(f"{idx+1}) {layer_vars.name}\n{layer_vars.numpy()}")


