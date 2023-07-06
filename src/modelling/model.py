import os
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras.activations import sigmoid, softplus
from keras.applications import VGG19, InceptionResNetV2
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Concatenate, Dense, Flatten, Input
from keras.losses import mean_absolute_error
from keras.models import Model
from keras.optimizers import AdamW
from keras.src.engine.keras_tensor import KerasTensor
from keras_vggface.vggface import VGGFace
from omegaconf import DictConfig
from tensorflow_probability.python.distributions import Distribution, Poisson
from tensorflow_probability.python.layers import DistributionLambda
from typing_extensions import Self

from preprocessing import Dataset, TrainValTest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def negloglik(y: tf.Tensor, p_y: Distribution) -> tf.Tensor:
    negloglik: tf.Tensor = -p_y.log_prob(y)

    return negloglik


class PoissonAgePredictor:
    def __init__(self: Self,
                 data: Dataset,
                 config: DictConfig) -> Self:
        # Attributes
        self.x: TrainValTest = data.x
        self.y_age: TrainValTest = data.y_age
        self.y_gender: TrainValTest = data.y_gender
        self.y_race: TrainValTest = data.y_race

        # Model definition
        input_shape: tuple = self.x.train.shape[1:]
        input: KerasTensor = Input(shape=input_shape, name='input')

        vgg19: KerasTensor = VGG19(input_shape=input_shape,
                                   include_top=False,
                                   weights='imagenet')(input)
        vgg19.trainable = False

        flatten: KerasTensor = Flatten(name='flatten')(vgg19)
        # fc1: KerasTensor = Dense(2048, activation="relu", name="fc1")(flatten)
        # fc2: KerasTensor = Dense(2048, activation="relu", name="fc2")(fc1)

        rate: KerasTensor = Dense(
            units=1,
            activation=softplus,
            name='rate'
        )(flatten)
        output: KerasTensor = DistributionLambda(
            lambda t: Poisson(rate=t,
                              validate_args=True,
                              allow_nan_stats=False),
            convert_to_tensor_fn=Distribution.mode,
            name='output'
        )(rate)

        # output: KerasTensor = Dense(units=1,
        #                             activation=softplus,
        #                             name='output')(flatten)

        self.model: Model = Model(
            inputs=input,
            outputs=output
        )

        self.model.compile(
            optimizer=AdamW(
                learning_rate=1e-4
            ),
            loss=negloglik
            # loss=mean_absolute_error
        )

    def fit(self: Self) -> None:
        self.history = self.model.fit(
            x=self.x.train,
            y=self.y_age.train,
            batch_size=64,
            validation_data=[self.x.val, self.y_age.val],
            epochs=10,
            callbacks=[ReduceLROnPlateau()]
        )

    def predict(self: Self,
                x_pred: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        y_pred = self.model.predict(x_pred)

        return y_pred
