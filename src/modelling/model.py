import math
import os

import tensorflow as tf
from keras.applications import VGG16
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input, Rescaling
from keras.losses import mean_absolute_error
from keras.models import Model
from keras.optimizers import AdamW
from omegaconf import DictConfig
from tensorflow_probability.python.distributions import Distribution, Poisson
from tensorflow_probability.python.layers import DistributionLambda
from typing_extensions import Self

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def negloglik(y: tf.Tensor, p_y: Distribution) -> tf.Tensor:
    negloglik: tf.Tensor = -p_y.log_prob(y)

    return negloglik


class AgePredictor:
    def __init__(self, input_shape: tuple[int, int]):
        self.input_shape: tuple[int, int] = input_shape

        # Define model
        self._input_ = Input(shape=input_shape, name='input')
        self._rescaling = Rescaling(scale=1. / 255, name='rescaling')(self._input_)
        self._vgg16 = VGG16(input_shape=input_shape,
                            include_top=False,
                            weights='imagenet')(self._rescaling)
        self._vgg16.trainable = False
        self._flatten = Flatten(name='flatten')(self._vgg16)
        self._fc1 = Dense(4096, activation="relu", name="fc1")(self._flatten)
        self._fc2 = Dense(4096, activation="relu", name="fc2")(self._fc1)
        self._output = Dense(units=1, name='output')(self._fc2)

        self.model = Model(
            inputs=self._input_,
            outputs=self._output
        )

        self.model.compile(
            optimizer=AdamW(
                learning_rate=1e-4,
                weight_decay=0.004
            ),
            loss=mean_absolute_error
        )

    def fit(self, training_data, validation_data, n_epochs):
        self.top_history = self.model.fit(
            x=training_data,
            batch_size=64,
            validation_data=validation_data,
            epochs=n_epochs,
            callbacks=[ReduceLROnPlateau(patience=3)]
        )

        self._vgg16.trainable = True
        self.vgg16_history = self.model.fit(
            x=training_data,
            batch_size=64,
            validation_data=validation_data,
            epochs=math.ceil(n_epochs / 2)
        )

    def evaluate(self, test_data):
        evaluation = self.model.evaluate(
            x=test_data,
            batch_size=64
        )

        return evaluation

    def predict(self, x):
        y_pred = self.model.predict(x)

        return y_pred
