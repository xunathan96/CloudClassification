from transforms import *
from utils import *
from dataset import *
from catalyst.dl.utils import load_checkpoint

NETWORK = 'Unet'
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DEVICE = 'cuda'

def initialize_model(network='Unet', type='train'):
    train, sub, train_ids, valid_ids, test_ids = load_data()

    model = None
    if network=='Unet':
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            activation=ACTIVATION,
            classes=4
        )
    elif network=='FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            activation=ACTIVATION,
            classes=4
        )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    data_ids = {
        'train': train_ids,
        'valid': valid_ids,
        'test': test_ids
    }
    dataset = initialize_dataset(train, sub, data_ids, preprocessing_fn)
    loaders = initialize_loaders(dataset, type)

    return model, dataset, loaders


def load_model(network, model_path):
    model = None
    if network=='Unet':
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            activation=ACTIVATION,
            classes=4
        )
    elif network=='FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            activation=ACTIVATION,
            classes=4
        )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    model.to(DEVICE)
    model.eval()

    checkpoint = load_checkpoint(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
