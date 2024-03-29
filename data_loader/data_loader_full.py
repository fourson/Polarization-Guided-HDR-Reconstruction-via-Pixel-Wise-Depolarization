from torchvision import transforms
from torch.utils.data import DataLoader

from .dataset import dataset_full
from base.base_data_loader import BaseDataLoader


class TrainDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset_full.TrainDataset(data_dir, transform=transform)

        super(TrainDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class InferDataLoader(DataLoader):
    def __init__(self, data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset_full.InferDataset(data_dir, transform=transform)

        super(InferDataLoader, self).__init__(self.dataset)


# for real data
class InferRealDataLoader(DataLoader):
    def __init__(self, data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset_full.InferRealDataset(data_dir, transform=transform)

        super(InferRealDataLoader, self).__init__(self.dataset)
