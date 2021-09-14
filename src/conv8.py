import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, edgeitems=10, linewidth=180)

# DNN parameters used in this example
# 5x5 input plane
input_height = 5
input_width = 5
# 1 channel input
channels_in = 2
# 2 chnnel output
channels_out = 1
# 3x3 kernel size
kernel_size = 3
# stride of 1
stride = 1

# conv layer weights initializer is:
#  [1, 2, ..., kernel_size * kernel_size * channels_in * channels_out]
kernel_weights_num = kernel_size * kernel_size * channels_in * channels_out
conv_weights = np.reshape(np.linspace(start=1,
                                      stop=kernel_weights_num,
                                      num=kernel_weights_num),
                          (channels_out, channels_in, kernel_size, kernel_size))

# conv layer bias initializer is array [channels_out, channels_out-1, ..., 1]
conv_bias = np.linspace(start=channels_out, stop=1, num=channels_out)

# Conv layer weights are in Height, Width, Input channels, Output channels (HWIO) format
conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

# generate input data
input_shape = (1, input_height, input_width, channels_in)
input_size = input_height * input_width * channels_in
x = np.reshape(np.linspace(start = input_size , stop = 1, num = input_size),
               (1, channels_in, input_height, input_width))

# input data is in Batch, Height, Width, Channels (BHWC) format
x = np.transpose(x, [0, 2, 3, 1])


# Create sequential model with 1 conv layer
# Conv layer has bias, no activation stride of 1, 3x3 kernel, zero input padding
# for output to have same dimension as input
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=channels_out,
                                 kernel_size=kernel_size,
                                 strides=stride,
                                 activation=None,
                                 use_bias=True,
                                 data_format="channels_last",
                                 padding="same",
                                 weights=[conv_weights, conv_bias]))

# Builds the model based on input shapes received
model.build(input_shape=input_shape)

# Use MSE for loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# print input data in BCHW format
print(f"input x:\n{np.squeeze(np.transpose(x, [0, 3, 1, 2]))}")

# Print Conv kernel in OIHW format
print(f"conv kernel weights:\n "\
      f"{np.transpose(model.trainable_variables[0].numpy(), [3, 2, 0, 1])}")

# print Conv bias
print(f"conv kernel bias: {model.trainable_variables[1].numpy()}")

# Create expected output
y_true = np.ones(shape=(1, input_height, input_width, channels_out))

# SGD update rule for parameter w with gradient g when momentum is 0:
# w = w - learning_rate * g
# For simplicity make learning_rate=1.0
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.0)

# Get model output y for input x, compute loss, and record gradients
with tf.GradientTape(persistent=True) as tape:
    xt = tf.convert_to_tensor(x)
    tape.watch(xt)
    y = model(xt)
    loss = loss_fn(y_true, y)


# dloss_dy is error gradient w.r.t. DNN output y
dloss_dy = tape.gradient(loss, y)

# dloss_dx is error gradient w.r.t DNN input x
dloss_dx = tape.gradient(loss, xt)

# Update DNN weights with gradients
grad = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grad, model.trainable_variables))

# print model output in BCHW format
print(f"output y:\n{np.squeeze(np.transpose(y, [0, 3, 1, 2]))}")

# print loss
print("loss: {}".format(loss))

# print dloss_dy: error gradient w.r.t. DNN output y, in BCHW format
print("dloss_dy:\n{}".format(np.squeeze((np.transpose(dloss_dy, [0, 3, 1, 2])))))

# print dloss_dx: error gradient w.r.t DNN input x, , in BCHW format
print("dloss_dx:\n{}".format(np.squeeze(np.transpose(dloss_dx, [0, 3, 1, 2]))))

# print updated conv layer kernel and bias weights
print(f"updated conv kernel:\n "\
      f"{np.transpose(model.trainable_variables[0].numpy(), [3, 2, 0, 1])}")
print(f"updated conv bias: {model.trainable_variables[1].numpy()}")
