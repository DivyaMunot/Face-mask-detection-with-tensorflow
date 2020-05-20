# Face-mask-classification-with-tensorflow

This project aims to detect whether or not a person is wearing a face mask.The image input which you give to the system will be analyzed
and the predicted result will be given as output.

## Conceptual Framework:
The project is entirely implemented using Python3.

The Conceptual Framework involved is mainly:
Keras – Tensorflow backend

## Model
I have built a model consisting of 5 convolutions and 5 max pooling layers, with ReLU as the activation function.

```python
model.summary()
```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_10 (Conv2D)           (None, 148, 148, 16)      448       
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 74, 74, 16)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 72, 72, 32)        4640      
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 36, 36, 32)        0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 34, 34, 64)        18496     
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 17, 17, 64)        0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 15, 15, 64)        36928     
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 5, 5, 64)          36928     
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 2, 2, 64)          0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 512)               131584    
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 513       
=================================================================
Total params: 229,537
Trainable params: 229,537
Non-trainable params: 0
_________________________________________________________________

## Accuracy
I achieved the training accuracy of 96.47%.
