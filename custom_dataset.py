import torch
from torchvision import ops
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import DataUtils, Visualizer
from PIL import Image
import random
import os


class CustomDataset(Dataset):
    def __init__(self, data_path: str, image_size: tuple = (416, 416), normalize: bool = False):
        self.data_path = data_path
        self.image_size = image_size
        self.normalize = normalize

        self.images_path = os.path.join(self.data_path, 'images')
        self.annotations_path = os.path.join(self.data_path, 'annotations')
        self.data = self._get_data()
        self.n_class = len(torch.unique(self.data['labels'])) - 1

        # Transforms
        self.transform_list = [
            transforms.ToTensor(),
            transforms.Resize(self.image_size)
        ]
        if self.normalize:
            self.transform_list = self.transform_list + [transforms.Normalize((0,), (255,))]
        self.transform = transforms.Compose(self.transform_list)

    def visualize_images(self, num_of_image, is_save_img=False):
        viz_path = DataUtils.create_custom_dir(directory_path='visualization')
        for idx in range(num_of_image):
            image, labels, bboxes = self.__getitem__(idx=idx)
            image = image * 255
            image = image.permute(1, 2, 0)
            Visualizer.visualize_data(image=image, bboxes=bboxes, is_save_img=is_save_img, viz_path=viz_path)

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.data['images'][idx])
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_size = image.size
        image = self.transform(image)

        # Transform bboxes for Scale, etc...
        bboxes = self._transform_boxes(bboxes=self.data['bboxes'][idx], orig_img_size=image_size)
        return [image, self.data['labels'][idx], bboxes]

    def _get_data(self) -> dict:
        # TODO: Augmentation
        # TODO: Sample Balancing
        data = {'images': os.listdir(self.images_path)}
        annotations_file_list = os.listdir(self.annotations_path)

        data_labels, data_bboxes = self._read_data_from_file(file_list=annotations_file_list)

        data['labels'] = pad_sequence(data_labels, batch_first=True, padding_value=-1)
        data['bboxes'] = pad_sequence(data_bboxes, batch_first=True, padding_value=-1)
        return data

    def _read_data_from_file(self, file_list):
        data_labels, data_bboxes = [], []
        for annot_file in file_list:
            annot_file_path = os.path.join(self.annotations_path, annot_file)
            file_labels, file_bboxes = [], []
            annotations = DataUtils.load_annotations_text_file(file_path=annot_file_path)
            for annot in annotations:
                annot_split = annot.split(' ')
                file_labels.append(int(annot_split[0]))
                file_bboxes.append([float(n) for n in annot_split[1:]])
            data_labels.append(torch.tensor(file_labels))
            data_bboxes.append(torch.tensor(file_bboxes))
        return data_labels, data_bboxes

    def _transform_boxes(self, bboxes: torch.tensor, orig_img_size: tuple) -> torch.tensor:
        transformed_box_data = []
        w_scale, h_scale = (self.image_size[0] / orig_img_size[0]), (self.image_size[1] / orig_img_size[1])
        for bboxes in bboxes:
            xc = bboxes[0] * w_scale
            yc = bboxes[1] * h_scale
            w = bboxes[2] * w_scale
            h = bboxes[3] * h_scale
            transformed_box_data.append([xc, yc, w, h])
        return torch.tensor(transformed_box_data)
