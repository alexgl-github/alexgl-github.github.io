import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import numpy as np

# number of integers in the input array
num_values = 5

# Custom layer to return index of the maximum value in the array
class Sample_Softmax(Layer):

    def __init__(self, num_values=5, tau=1.0):
        super(Sample_Softmax, self).__init__()
        self.tau = tau
        self.arange = tf.range(start=0, limit=num_values, dtype=tf.float32)

    def call(self, x):
        # generate  gumbel noise
        noise = np.random.gumbel(size=x.shape)
        noisy_x = x + noise

        # apply softmax temperature
        noisy_x =  noisy_x / self.tau

        # compute softmax
        probs = K.exp(noisy_x) / K.sum(K.exp(noisy_x))

        # dot product for the index of max array element
        index = tf.tensordot(probs, self.arange, axes=1)

        return index

# toy model with one Dense layer and one custom sample layer
model = tf.keras.Sequential()
model.add(Dense(num_values, activation=None, use_bias=False, kernel_initializer=tf.keras.initializers.zeros()))
model.add(Sample_Softmax(tau=0.75))

# loss function is L1 distance between predicted and actual index
def loss_fn(target, x_pred):
    x_pred = tf.cast(x=x_pred, dtype=tf.float32)
    return tf.math.reduce_sum(tf.math.abs(x_pred - target))

# optimizer with 0.5 learning rate
optimizer=RMSprop(lr=0.5)

# Model traininig input is shuffled array of numbers
x = np.arange(num_values).astype(np.float32)
np.random.shuffle(x)

# add batch axis
x = x[np.newaxis, ...]

# Model label is index of the maximum value in the array
y = np.argmax(x)

# Training loop
for idx in range(4):
    with tf.GradientTape() as tape:
        pred = model(x)[0]
        loss = loss_fn(y, pred)
        grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    print(f"Training x={x} tgt={y} pred={pred:.1f} loss={loss:.2f}")

# Test with the same data used for training
pred = model(x)[0]
print(f"Testing x={x} expected y={y} pred={pred:.1f}")
