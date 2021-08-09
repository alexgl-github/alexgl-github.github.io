import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, edgeitems=10, linewidth=180)

input_height = 5
input_width = 5
kernel_size = 3
channels_in = 1
channels_out = 1
strides = 1
# Create sequential model
model = tf.keras.Sequential()

dense_weights_num = input_height*input_width * input_height*input_width
dense_weights = np.reshape(np.linspace(start=1, stop=dense_weights_num, num=dense_weights_num), (input_height * input_width, input_height * input_width))
kernel_weights_num = kernel_size * kernel_size * channels_in * channels_out
conv_weights = np.reshape(np.linspace(start=1, stop=kernel_weights_num, num=kernel_weights_num), (kernel_size, kernel_size, channels_in, channels_out))

#biases = np.ones(shape=(channels_out,))

#model.add(tf.keras.layers.Flatten())
#/model.add(tf.keras.layers.Dense(input_height * input_width * channels_in, use_bias=False, weights=[dense_weights]))
#model.add(tf.keras.layers.Reshape((input_height,input_width, channels_in)))
model.add(tf.keras.layers.Conv2D(filters=channels_out,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 activation=None,
                                 use_bias=False,
                                 data_format="channels_last",
                                 padding="same",
                                 name="conv1",
                                 weights=[conv_weights]))


# use MSE as loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Arbitrary model input
#x = np.reshape(np.linspace(start=1, stop=input_height * input_width, num=input_height * input_width), (input_height, input_width, 1))
x = np.ones(shape=(input_height, input_width, channels_in))
x = x[np.newaxis, ...]

y=model(x)
model.summary()

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
    xt = tf.convert_to_tensor(x)
    tape.watch(xt)
    y = model(xt)

    # obtain MSE loss
    loss = loss_fn(y_true, y)

print("initial weights:")
#print(f"dense weights=\n {model.trainable_variables[0].numpy()}")
print(f"conv weights=\n {np.transpose(model.trainable_variables[0].numpy(), [2, 3, 0, 1])}")
#print(f"conv bias=\n {model.trainable_variables[3].numpy()}")

# adjust Dense layer weights
dloss_dy = tape.gradient(loss, y)
dloss_dy = np.transpose(dloss_dy.numpy(), [1, 2, 3, 0])
dloss_dx = tape.gradient(loss, xt)
grad = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grad, model.trainable_variables))

# print model input and output excluding batch dimention
print(f"x={x.shape} y={y.shape} y_true={y_true.shape} loss={loss.shape} dloss_dy={dloss_dy.shape}")

print(f"input x=\n{np.squeeze(x)}")
print(f"output y=\n{np.squeeze(y)}")
print(f"expected output y_true=\n{np.squeeze(y_true)}")

print("dloss_dy=\n{}".format(np.squeeze(dloss_dy)))
#print(f"dw[0]=\n{np.squeeze(grad[0])}")
print(f"dw[0]=\n{np.squeeze(grad[0])}")
c = tf.nn.conv2d(x, dloss_dy, strides=1, padding=[(0,0), (1,1), (1,1), (0,0)])
print(f" (x,dloss_dy)=\n{np.squeeze(c)}")
print("dloss_dx=\n{}".format(np.squeeze(dloss_dx)))

# print MSE loss
print(f"loss={loss}")

# print updated dense layer weights
print("updated weights=")
#print(f"dense weights=\n {model.trainable_variables[0].numpy()}")
print(f"conv weights=\n {np.transpose(model.trainable_variables[0].numpy(), [2, 3, 0, 1])}")
#print(f"conv bias=\n {model.trainable_variables[3].numpy()}")


