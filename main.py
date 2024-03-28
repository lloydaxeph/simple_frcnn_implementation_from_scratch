from custom_dataset import CustomDataset
from models import CustomObjectDetector

from PIL import Image
import torch
import cv2
import os


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # PARAMETERS -------------------------------------------------------------------------------------------------------
    DATA_PATH = r'C:\Users\Lloyd Acha\Documents\ACHA_Files\Projects\2024\Data'
    mode = 'test_video'  # train, fine_tune, test_images, test_video
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

    load_model_path = os.path.join("models", "model_02062024_50e_1500d",
                                   'best_model - Copy.pt')

    video_path = r'C:\Users\Lloyd Acha\Documents\ACHA_Files\Projects\2024\Data\Goats_Custom_DS\video'
    video_name = 'goat_test1.mp4'
    full_video_path = os.path.join(video_path, video_name)
    # -----------------------------------------------------------------------------------------------------------------
    assert mode in ['train', 'fine_tune', 'test_images', 'test_video']
    detector = CustomObjectDetector(train_data=train_ds, test_data=test_ds, val_data=test_ds,
                                    early_stopping_patience=early_stopping_patience,
                                    anc_scales=anc_scales,
                                    anc_ratios=anc_ratios)
    detector.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)