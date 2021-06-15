import tensorflow as tf
import numpy as np
import importlib
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

class f1(object):

    def __init__(self):
        self.preds = np.zeros((0, 0))
        self.eps = 1e-10

    def update(self, y_true, y_pred):
        for y_t, y_p in zip(y_true, y_pred):
            if self.preds.shape[1] <= y_t:
                self.resize(y_t+1)
            if self.preds.shape[0] <= y_p:
                self.resize(y_p+1)
            self.preds[y_p, y_t] += 1

    def update_onehot(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
        for y_t, y_p in zip(y_true, y_pred):
            if self.preds.shape[1] <= y_t:
                self.resize(y_t+1)
            if self.preds.shape[0] <= y_p:
                self.resize(y_p+1)
            self.preds[y_p, y_t] += 1

    def resize(self, new_size):
        inc = new_size - self.preds.shape[0]
        row = np.zeros((inc, self.preds.shape[1]))
        self.preds = np.concatenate((self.preds, row), axis=0)
        col = np.zeros((self.preds.shape[0], inc))
        self.preds = np.concatenate((self.preds, col), axis=1)

    def nclasses(self):
        return len(self.preds)

    def score(self):
        scores = []
        for idx in range(self.nclasses()):
            tp = self.preds[idx, idx]
            fp_tp = np.sum(self.preds[idx, :])
            fn_tp = np.sum(self.preds[:, idx])
            precision = (tp) / (fp_tp + self.eps)
            recall = (tp) / (fn_tp + self.eps)
            score = 2 * precision * recall / (precision + recall + self.eps)
            scores.append(score)

        return np.mean(scores)

    def reset(self):
        self.preds = np.zeros((0, 0))

def get_dataset(data):
    dataset_length = len(data)
    def representative_dataset():
        for _ in range(dataset_length):
            idx = np.random.randint(low=0, high=dataset_length)
            x = data[idx]
        yield [x]
    return representative_dataset

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='sigmoid', kernel_initializer=tf.keras.initializers.zeros(), bias_initializer=tf.keras.initializers.zeros()),
    tf.keras.layers.Dense(10, activation=None, kernel_initializer=tf.keras.initializers.zeros(), bias_initializer=tf.keras.initializers.zeros()),
    tf.keras.layers.Softmax()
])

learning_rate = 0.01
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data("./mnist.npz")
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

loss_epoch = 0.0
f1_score = f1()
iterations_epoch = 1000 #len(x_train)

for iter in range(iterations_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(x_train[0:0+1])
        y_true = y_train[0:0+1]
        loss = loss_fn(y_true, y_pred)
        gradients = tape.gradient([loss], model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_epoch += loss
    f1_score.update_onehot(y_true=y_true, y_pred=y_pred)
    #if (iter % 500) == 0:
    print(f"y_pred={y_pred} y_true={y_true}")
    print(f"{iter}/{iterations_epoch} loss={loss} {loss_epoch/(iter+1)} f1={f1_score.score()}")


    if iter==1:
        #print("f{iter} updated weights=\n")
        #for idx, layer_vars in enumerate(model.trainable_variables):
        #    print(f"{idx+1}) {layer_vars.name}\n{layer_vars.numpy()}")
        pass

exit(0)
print(f"loss={loss_epoch/iterations_epoch} f1={f1_score.score()}")

model.save("./mnist")
model.summary()
converter = tf.lite.TFLiteConverter.from_saved_model("./mnist")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = get_dataset(x_train)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open("mnist.tflite", "wb") as fd:
    fd.write(tflite_quant_model)


