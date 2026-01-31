"""
This package includes all the modules related to data loading and preprocessing
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import os

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py"."""
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    """Create a dataset given the option."""
    dataset_class = find_dataset_using_name(opt.dataset_mode)
    dataset = dataset_class(opt)
    print("dataset [%s] was created" % type(dataset).__name__)

    sampler = None
    shuffle = not opt.serial_batches

    # In DDP mode, create a DistributedSampler
    if "LOCAL_RANK" in os.environ:
        print(f'create DDP sampler on rank {int(os.environ["LOCAL_RANK"])}')
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        # When using DistributedSampler, shuffle argument must be False in DataLoader
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        pin_memory=True,
        drop_last=True  # Recommended for DDP to avoid uneven batches
    )

    # Add a method to the dataloader to set the epoch for the sampler
    def set_epoch(epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)
    
    dataloader.set_epoch = set_epoch

    return dataloader