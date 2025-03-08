import sys, os, errno, signal, copy
from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset
import yaml
from omegaconf import OmegaConf
import musicnet
from time import time

# Load config.yaml using OmegaConf for structured access
config = OmegaConf.load("config.yaml")

# Ensure checkpoint directory exists
os.makedirs(config.get("checkpoint_path", "./checkpoints"), exist_ok=True)

# Set CUDA configurations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu.device", "0"))




def load_music_dataset(dataset_config):
    """Loads dataset from tokens stored in a directory"""
    dict_ds = {"input_ids": []}
    for file in os.listdir(dataset_config.root):
        d = torch.squeeze(torch.load(os.path.join(dataset_config.root, file)))
        dict_ds["input_ids"].append(d)
    return Dataset.from_dict(dict_ds)


def group_texts(examples, block_size):
    """Groups text into chunks of block_size"""
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    return {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }


def get_dataset(config, mode, num_proc=8):
    """Loads dataset based on config parameters"""
    tokenized_dataset = load_music_dataset(config)
    chunked_dataset = tokenized_dataset.map(
        lambda x: group_texts(x, config.window),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )
    return chunked_dataset.with_format("torch")


def get_dataloaders(config, distributed=True):
    """Creates train and validation dataloaders"""

    # Load datasets
    train_set = get_dataset(config.data.train, "train")
    test_set = get_dataset(config.data.test, "test")

    if distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(test_set)
    else:
        train_sampler = None
        test_sampler = None

    # Create data loaders
    train_loader = cycle_loader(
        DataLoader(
            train_set,
            batch_size=config.data.train.batch_size,
            sampler=train_sampler,
            num_workers=config.data.train.num_workers,
            pin_memory=config.data.train.pin_memory,
            shuffle=config.data.train.shuffle and train_sampler is None,
            persistent_workers=True,
        )
    )

    test_loader = cycle_loader(
        DataLoader(
            test_set,
            batch_size=config.data.test.batch_size,
            sampler=test_sampler,
            num_workers=config.data.test.num_workers,
            pin_memory=config.data.test.pin_memory,
            shuffle=config.data.test.shuffle and test_sampler is None,
        )
    )

    return train_loader, test_loader



def worker_init(args):
    """Worker initialization for DataLoader"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore signals so parent can handle them
    np.random.seed(os.getpid() ^ int(time()))  # Set unique seed per worker


def get_musicnet_dataset(config, train=True):
    """Loads the MusicNet dataset based on the configuration"""
    dataset_config = config.data.train if train else config.data.test

    return musicnet.MusicNet(
        root=dataset_config.root,
        train=dataset_config.train,
        download=dataset_config.download,
        window=dataset_config.window,
        epoch_size=dataset_config.get("epoch_size", 50000) if not train else None,
    )

def cycle_loader(dataloader, sampler=None):
    """Infinite data loader cycle"""
    while True:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data



def get_dataloaders(config, distributed=True):
    """Creates cyclic train & validation dataloaders"""

    # Load datasets
    train_set = get_musicnet_dataset(config, train=True)
    test_set = get_musicnet_dataset(config, train=False)

    if distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(test_set)
    else:
        train_sampler = None
        test_sampler = None

    # DataLoader Arguments
    loader_kwargs = {
        "num_workers": config.data.train.num_workers,
        "pin_memory": config.data.train.pin_memory,
        "worker_init_fn": worker_init,
    }

    # Create data loaders
    train_loader = cycle_loader(
        DataLoader(
            train_set,
            batch_size=config.data.train.batch_size,
            sampler=train_sampler,
            shuffle=config.data.train.shuffle if train_sampler is None else False,
            **loader_kwargs,
        )
    )

    test_loader = cycle_loader(
        DataLoader(
            test_set,
            batch_size=config.data.test.batch_size,
            sampler=test_sampler,
            shuffle=config.data.test.shuffle if test_sampler is None else False,
            **loader_kwargs,
        )
    )

    return train_loader, test_loader
