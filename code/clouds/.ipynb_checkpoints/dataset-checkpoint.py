from utils import *
from transforms import *

class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms = albu.Compose([albu.HorizontalFlip(),AT.ToTensor()]),
                preprocessing=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)


def initialize_dataset(train, test, data_ids, preprocessing_fn):
    train_dataset = CloudDataset(
        df=train,
        datatype='train',
        img_ids=data_ids['train'], #train_ids,
        transforms=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(
        df=train,
        datatype='valid',
        img_ids=data_ids['valid'], #valid_ids,
        transforms=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn))
    test_dataset = CloudDataset(
        df=test,
        datatype='test',
        img_ids=data_ids['test'], #test_ids,
        transforms=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn))

    dataset = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}
    return dataset


def initialize_loaders(dataset, type):
    num_workers = 0
    bs = 16

    train_loader = DataLoader(
        dataset['train'],
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers)
    valid_loader = DataLoader(
        dataset['valid'],
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers)
    test_loader = DataLoader(
        dataset['test'],
        batch_size=8,
        shuffle=False,
        num_workers=0)

    loaders = None
    if type == 'train':
        loaders = {
            "train": train_loader,
            "valid": valid_loader
        }
    elif type == 'infer':
        loaders = {
            "infer": valid_loader
        }
    elif type == 'test':
        loaders = {
            "test": test_loader
        }

    return loaders
