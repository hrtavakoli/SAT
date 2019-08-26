
MODEL_NAME = ['deepgaze2', 'mlnet', 'resnetsal', 'salicon', 'samres']


class ModelConfig:

    MODEL = 'samres'
    B_SIZE = 10
    N_STEP = 3
    W_OUT = 40
    H_OUT = 30


def make_model(meta_config):
    ''' make the model from the meta config'''

    m = __import__(meta_config.MODEL)
    model = getattr(m, 'Model')

    if meta_config.MODEL == 'samres':
        object = model(meta_config.B_SIZE, meta_config.N_STEP, meta_config.W_OUT, meta_config.H_OUT)
    else:
        object = model()

    return object



if __name__ == '__main__':
    meta_config = ModelConfig
    a = make_model(meta_config)
    print(a)
