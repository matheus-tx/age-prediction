import os

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras.activations import softplus
from keras.applications import VGG19, EfficientNetV2L
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import AdamW
from keras.src.engine.keras_tensor import KerasTensor
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
                 train_set: Dataset,
                 #  val_set,
                 #  test_set,
                 dataset,
                 config: DictConfig) -> Self:
        # Attributes
        self.train_set = train_set
        # self.val_set = val_set
        # self.test_set = test_set

        self.dataset = dataset

        # Model definition
        input_shape: tuple = self.train_set.element_spec[0].shape[1:]
        input_: KerasTensor = Input(shape=input_shape, name='input')

        vgg19: KerasTensor = VGG19(input_shape=input_shape,
                                   include_top=False,
                                   weights='imagenet')(input_)
        vgg19.trainable = False
        flatten: KerasTensor = Flatten(name='flatten')(vgg19)
        fc1: KerasTensor = Dense(2048, activation="relu", name="fc1")(flatten)
        fc2: KerasTensor = Dense(2048, activation="relu", name="fc2")(fc1)
        rate: KerasTensor = Dense(
            units=1,
            activation=softplus,
            name='rate'
        )(fc2)
        output: KerasTensor = DistributionLambda(
            lambda t: Poisson(rate=t,
                              validate_args=True,
                              allow_nan_stats=False),
            convert_to_tensor_fn=Distribution.mode,
            name='output'
        )(rate)
        self.model: Model = Model(
            inputs=input_,
            outputs=output
        )

        # Compile model
        self.model.compile(
            optimizer=AdamW(
                learning_rate=1e-4
            ),
            loss=negloglik
        )

    def fit(self: Self) -> None:
        self.history = self.model.fit(
            self.train_set,
            # steps_per_epoch=self.train_set.n // self.train_set.batch_size,
            # validation_data=self.val_set,
            # validation_steps=self.val_set.n // self.val_set.batch_size,
            # epochs=10,
            # callbacks=[ReduceLROnPlateau()]
        )

        # self.history = self.model.fit(
        #     x=self.dataset.x.train,
        #     y=self.dataset.y_age.train,
        #     validation_data=[self.dataset.x.val, self.dataset.y_age.val],
        #     epochs=10,
        #     callbacks=[ReduceLROnPlateau()],
        #     batch_size=64
        # )

    def predict(self: Self,
                x_pred: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        y_pred = self.model.predict(x_pred)

        return y_pred
