from custom_dataset import CustomDataset
from models import CustomObjectDetector

import torch
import os


def define_custom_dataset(train_data_path: str, test_data_path: str, val_data_path: str, image_size: tuple,
                          normalize: bool) -> (CustomDataset, CustomDataset, CustomDataset):
    """EXAMPLE CUSTOM DATASET IMPLEMENTATION"""
    train_ds = CustomDataset(data_path=train_data_path, image_size=image_size, normalize=normalize)
    test_ds = CustomDataset(data_path=test_data_path, image_size=image_size, normalize=normalize)
    val_ds = CustomDataset(data_path=val_data_path, image_size=image_size, normalize=normalize)
    return train_ds, test_ds, val_ds


def train_model(datasets: tuple, epochs: int, batch_size: int, learning_rate: float, early_stopping_patience: int,
                anc_scales: tuple, anc_ratios: tuple) -> CustomObjectDetector:
    train_ds, test_ds, val_ds = datasets
    """MODEL TRAIN EXAMPLE"""
    detector = CustomObjectDetector(train_data=train_ds, test_data=test_ds, val_data=test_ds,
                                    early_stopping_patience=early_stopping_patience,
                                    anc_scales=anc_scales,
                                    anc_ratios=anc_ratios)
    detector.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    return detector


def demo(detector: CustomObjectDetector, model_path: str, num_test_images: int):
    """MODEL DEMONSTRATION"""
    detector.load_model(model_path=model_path)
    detector.test_images(num_images=num_test_images)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # PARAMETERS -------------------------------------------------------------------------------------------------------
    PROJECT_PATH = '.../project'
    train_data_path = os.path.join(PROJECT_PATH, 'data', 'train')
    test_data_path = os.path.join(PROJECT_PATH, 'data', 'test')
    val_data_path = os.path.join(PROJECT_PATH, 'data', 'validation')
    model_path = os.path.join(PROJECT_PATH,'model', 'my_model.pt')

    image_size = (416, 416)
    normalize = True

    epochs = 3
    batch_size = 8
    learning_rate = 1e-3
    early_stopping_patience = 0
    anc_scales = (2, 4, 6)
    anc_ratios = (0.5, 1, 1.5)

    num_test_images = 5

    # EXAMPLE IMPLEMENTATION -------------------------------------------------------------------------------------------
    datasets = define_custom_dataset(train_data_path=train_data_path, test_data_path=test_data_path,
                                     val_data_path=val_data_path, image_size=image_size, normalize=normalize)
    detector = train_model(datasets=datasets, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                           early_stopping_patience=early_stopping_patience, anc_scales=anc_scales,
                           anc_ratios=anc_ratios)
    demo(detector=detector, model_path=model_path, num_test_images=num_test_images)
    # ------------------------------------------------------------------------------------------------------------------
