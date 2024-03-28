from custom_dataset import CustomDataset
from models import CustomObjectDetector

from PIL import Image
import torch
import cv2
import os


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # PARAMETERS -------------------------------------------------------------------------------------------------------

    #LOCAL FILES
    DATA_PATH = r'\Projects\Data'
    load_model_path = os.path.join("models", "model_02042024_50e_3336d",
                                   'best_model.pt')

    # Dataset parameters
    image_size = (416, 416)
    normalize = True
    dataset_name = 'Farms_DS'
    train_ds = CustomDataset(data_path=os.path.join(DATA_PATH, dataset_name, 'train'),
                             image_size=image_size,
                             normalize=normalize)
    test_ds = CustomDataset(data_path=os.path.join(DATA_PATH, dataset_name, 'test'),
                            image_size=image_size,
                            normalize=normalize)

    # Training parameters
    epochs = 3
    batch_size = 8
    learning_rate = 1e-3
    early_stopping_patience = 0
    anc_scales = [2, 4, 6]
    anc_ratios = [0.5, 1, 1.5]

    mode = 'test_images'  # train, fine_tune, test_images

    num_test_images = 5
    # -----------------------------------------------------------------------------------------------------------------
    assert mode in ['train', 'fine_tune', 'test_images']

    # we will just use the test_data as the validation data for testing purposes.
    detector = CustomObjectDetector(train_data=train_ds, test_data=test_ds, val_data=test_ds,
                                    early_stopping_patience=early_stopping_patience,
                                    anc_scales=anc_scales,
                                    anc_ratios=anc_ratios)

    if mode == 'train':
        print('Training Triggered.')
        detector.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    elif mode == 'fine_tune':
        detector.load_model(model_path=load_model_path)
        detector.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    elif mode == 'test_images':
        detector.load_model(model_path=load_model_path)
        detector.test_images(num_images=num_test_images)
