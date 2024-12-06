from datasets import load_dataset


def load_training_dataset(random_state=1):
    dataset = load_dataset('json', data_files='OBEDATA/OBEDATA-L-TRAIN.json', split='train')
    dataset = dataset.shuffle(seed=random_state)
    return dataset


def load_validation_dataset():
    return load_dataset('json', data_files='OBEDATA/OBEDATA-L-DEV.json', split='train')


def load_test_dataset():
    return load_dataset('json', data_files='OBEDATA/OBEDATA-L-DEV.json', split='train')
