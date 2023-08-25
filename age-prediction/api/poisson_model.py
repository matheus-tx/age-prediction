from keras.activations import relu, softplus
from keras.layers import Dense, Flatten, Input, Rescaling
from keras_vggface.vggface import VGGFace


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

    del rate
