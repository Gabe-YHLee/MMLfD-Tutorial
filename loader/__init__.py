from torch.utils import data
from loader.Pouring_dataset import Pouring
from loader.Toy_dataset import TwoDimToy

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader
        
def get_dataset(data_dict, **kwargs):
    name = data_dict["dataset"]
    if name == 'Pouring':
        dataset = Pouring(**data_dict)
    elif name == 'Toy':
        dataset = TwoDimToy(**data_dict)
    return dataset