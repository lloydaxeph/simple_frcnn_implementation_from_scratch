# Simple F-RCNN Implementation from Scratch

## 1.0 About
A simple demonstration on how to implement a <b>Faster Region-Based Convolutional Neural Network (F-RCNN)</b> from scratch using [pytorch](https://pytorch.org/).

<b>Basic summary of what the model does:</b>
1. Generate region proposals for each image.
2. Calculates the IOU (Intersection Over Union) of each proposed region (PR) vs the ground truth (GT) data.
3. Performs Transfer learning using the PRs with labels.
4. Performs classification on the PRs.

### 1.1 Two Stage Detector Model
To do this, we have to divide the model into 2 stages. The main part which is the <b>Region Proposal Network (RPN)</b>. And the other one which is the <b> Classification Module</b>.
<br/>
![Two Stage Detector Model Diagram](https://github.com/lloydaxeph/simple_frcnn_implementation_from_scratch/assets/158691653/6e01717f-e888-470d-90db-eeabe48ae341)<br/>

## 2.0 Getting Started
### 2.1 Installation
Install the required packages
```
pip3 install -r requirements.txt
```
### 2.2 Data Format
Data folder should have directory names <b>images</b> for the image <i>(.jpg/png)</i> files and <b>annotations</b> for the annotations file <i>(.txt)</i>.

<b>NOTE:</b> Each for each image file in the <i>images</i> director, there should be a corresponding annotations file with the same name in the <i>annotations</i> directory.
```
data_folder
│
└───images
│   │   image001.jpg
│   │   image002.jpg
│   │...
│
└───annotations
    │   image001.txt
    │   image002.txt
    │...
```

Annotations format would be similar to [YOLO's](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format) where the file should be formatted with one row per object in <b>class x_center y_center width height</b> format. However, the difference here is that the data is not normalized.
```
0 233.698 634.239 562.796 1023.359
1 311.171 151.679 759.360 1023.359
...
```
### 2.3 Create a Custom Dataset
You can create a <b>Custom Dataset</b> using the following code:

```python
train_ds = CustomDataset(data_path=train_data_path, image_size=image_size, normalize=normalize)
test_ds = CustomDataset(data_path=test_data_path, image_size=image_size, normalize=normalize)
val_ds = CustomDataset(data_path=val_data_path, image_size=image_size, normalize=normalize)
```
### 2.4 Train a Model
You can create and train a <b>CustomObjectDetector model</b> using the following code:
```python
detector = CustomObjectDetector(train_data=train_ds, 
                                test_data=test_ds, 
                                val_data=test_ds,
                                early_stopping_patience=early_stopping_patience,
                                anc_scales=anc_scales,
                                anc_ratios=anc_ratios)
detector.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
```
### 1.5 Use the Model
You can demonstrate how your trained model works by using the CustomObjectDetector's <b>test_images</b> function.
```python
detector.test_images(num_images=3)
```

For the complete demonstration, you can follow the [sample_implementation.py](https://github.com/lloydaxeph/simple_frcnn_implementation_from_scratch/blob/master/sample_implementation.py) script above.



