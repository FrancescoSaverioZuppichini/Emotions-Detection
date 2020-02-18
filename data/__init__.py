import numpy as np
from .FacialExpressionDataset import FacialExpressionDataset
from torch.utils.data import DataLoader, random_split
from logger import logging
from torch.utils.data import random_split, Subset

def get_dataloaders(
        data_dir,
        train_transform=None,
        val_transform=None,
        split=(0.5, 0.5),
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    # create the datasets, the train dataset need to be crated in order to use the train_transform
    train_ds = FacialExpressionDataset(root=data_dir, transform=train_transform)
    ds = FacialExpressionDataset(root=data_dir, transform=val_transform)
    # split the full ds in train and val
    val_size = int(len(ds) * split[0])
    train_size = len(ds) - val_size 
    train_ds_tmp, val_ds = random_split(ds, (train_size, val_size))
    # use the computed indices to create a subset from the train_ds, this will allow 
    # us to use the correct transformation
    train_ds = Subset(train_ds, train_ds_tmp.indices)
    # add train_transform
    train_ds.dataset.transform = train_transform
    # now we want to split the val_ds in validation and test
    test_size = int(val_size * split[1])
    val_size = val_size - test_size 
    val_ds, test_ds = random_split(val_ds, (val_size, test_size))
    test_ds.dataset.transform = val_transform
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl

