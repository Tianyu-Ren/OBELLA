import torch


def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')


def cpu():
    return torch.device("cpu")


def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return torch.cuda.device_count()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [gpu(i) for i in range(num_gpus())]
    if len(devices) == 0:
        devices = [cpu()]
    return devices


def data_loading(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size)


def generate_json_files(inputs):
    pass
