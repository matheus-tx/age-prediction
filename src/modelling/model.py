import os

import tensorflow as tf
from keras.applications import VGG19
from keras.layers import Concatenate, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import AdamW
from omegaconf import DictConfig
from tensorflow_probability.python.distributions import (Distribution,
                                                         NegativeBinomial)
from tensorflow_probability.python.layers import DistributionLambda
from typing_extensions import Self

from preprocessing import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def negloglik(y: tf.Tensor, p_y: Distribution) -> tf.Tensor:
    negloglik: tf.Tensor = -p_y.log_prob(y)

    return negloglik


class NegativeBinomialVGG19:
    def __init__(self: Self,
                 data: Dataset,
                 config: DictConfig) -> Self:
        self.x_picture = data.x_picture
        self.x_feature = data.x_feature
        self.y = data.y

        input_picture = Input(shape=self.x_picture.train.shape[1:],
                              name='input_picture')
        input_feature = Input(shape=self.x_feature.train.shape[1:],
                              name='input_feature')

        vgg19 = VGG19(input_shape=self.x_picture.train.shape[1:],
                      include_top=False,
                      weights='imagenet')(input_picture)
        vgg19.trainable = False

        x = Flatten(name='flatten')(vgg19)
        x = Dense(2048, activation="relu", name="fc1")(x)
        x = Dense(2048, activation="relu", name="fc2")(x)

        feature_vector: Concatenate = Concatenate(
            axis=-1,
            name='concat_params'
        )([x, input_feature])

        total_count: Dense = Dense(
            units=1,
            activation='softplus',
            name='total_count'
        )(feature_vector)
        prob: Dense = Dense(
            units=1,
            activation='sigmoid',
            name='prob'
        )(feature_vector)

        params: Concatenate = Concatenate(
            axis=-1,
            name='concat_params'
        )([total_count, prob])

        output: DistributionLambda = DistributionLambda(
            lambda t: NegativeBinomial(
                total_count=total_count,
                probs=prob,
                validate_args=True,
                allow_nan_stats=False
            ),
            convert_to_tensor_fn=Distribution.mode,
            name='output'
        )(params)

        self.model: Model = Model(
            input=[input_picture, input_feature],
            output=output
        )

        self.model.compile(
            optimizer=AdamW(),
            loss=negloglik
        )

    def fit(self) -> None:
        self.model.fit(
            x=[self.x_picture.train, self.x_feature.train],
            y=self.y.train,
            batch_size=64,
            validation_data=[[self.x_picture.val, self.x_feature.val],
                             self.y.val]
        )
