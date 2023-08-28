from keras.activations import relu, softplus
from keras.layers import Dense, Flatten, Input, Rescaling
from keras.models import Model
from keras_vggface.vggface import VGGFace
from tensorflow_probability.python.distributions import Distribution, Poisson
from tensorflow_probability.python.layers import DistributionLambda


def negloglik(y, p_y):
    negloglik = -p_y.log_prob(y)

    return negloglik


def make_poisson_model(input_shape):
    # Preprocessing
    input_ = Input(shape=input_shape, name='input')
    rescaling = Rescaling(scale=1. / 255, name='rescaling')(input_)

    vggface = VGGFace(
        input_shape=input_shape,
        include_top=False,
        model='resnet50'
    )(rescaling)

    flatten = Flatten(name='flatten')(vggface)
    fc1 = Dense(4096, activation=relu, name='fc1')(flatten)
    fc2 = Dense(4096, activation=relu, name='fc2')(fc1)
    rate = Dense(1, activation=softplus, name='rate')(fc2)

    output = DistributionLambda(
        lambda t: Poisson(
            rate=t,
            validate_args=True,
            allow_nan_stats=False,
            force_probs_to_zero_outside_support=True
        ),
        convert_to_tensor_fn=Distribution.mode,
        name='output'
    )(rate)

    model = Model(inputs=input_, outputs=output)

    return model
