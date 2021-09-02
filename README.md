# Pytorch_tutorial_object_detection_MNIST
A newbie-friendly playground to understand and experiment object-detection in Pytorch

Training object-detection models on standard datasets can be quite computationally intensive. Here, I generate an object-detection dataset with MNIST to help learn and experiment more on the topic. This work is intended for those who want to try object detection **with little computation resource**. It will be a summary and compilation of the tutorial(s) on github I went through, as well as other algorithms I want to try implementing in the future. The content will focus on making "a summary I wished there was when I was learning it".

For the purpose of easy experimentation and establishing a work flow, the main code is presented in Colab notebooks, with core sections (models definition, training, ...) defined explicitly inside.

Let's dive right in!👏
# Contents

[***Data***](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#data)

[***Single-shot detection (SSD)***](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#single-shot-detection-ssd)

  [Conventions](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#conventions)
  
  [Work flow: main.ipynb](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#work-flow-mainipynb)
  
  [Implementation details](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#implementation-details)
  
    [Base Convolution](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#base-convolution)
    
    [Auxiliary Convolution](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#auxiliary-convolution)
    
    [Prediction Convolution](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#prediction-convolution)
    
    [MultiBoxLoss](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#multiboxloss)
    
    [Dataset class](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#dataset-class)
    
    [Evaluation](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#evaluation)
    
  [Experiences learnt](https://github.com/PhanHuyThong/Pytorch_tutorial_object_detection_MNIST/blob/master/README.md#experiences-learnt)

# Data
Inspired by my friend's [work](https://github.com/nguyenvantui/mnist-object-detection/blob/master/mnist_gen.py).
See data_generation.ipynb in this repo for a cleaned notebook version.

## Generate data
1. Load MNIST
2. Generate object-detection data:

  Set image size h,w. I recommend `h = w = 32 pixels` for fast experiments later on. 
  
  **I assume `h = w` and refer to `image_size = h` from this point onwards**.
  
  `n_generate` = number of images to generate
  
  In generating each image:
  
   - Loop through some regions of the image and randomly put a *slightly augmented* MNIST digit into that region
    
   - Record bounding boxes and labels
    
3. Save the generated datasets (see /data/train.pkl)

----------------------------
/data/train.pkl (and same for test.pkl) is a tuple of `(img, boxes, labels)`, where

  * `img: list [n_generated]` of numpy arrays `[h,w], dtype=int, between [0, 255]`.  
  
  * `boxes: list [n_generated]` of lists `[n_objects, 4], dtype=int, between [0, h]`. Boxes in **absolute boundary coordinates**: `(x_min, y_min, x_max, y_max)`. Note that n_objects can vary for each image.
  
  * `labels: list [n_generated]` of lists `[n_objects], dtype=int`. The label for each box. 
  
Examples:
![image](https://user-images.githubusercontent.com/43468452/131778124-4981819e-c9f2-4f76-9149-094bb4f8e955.png)

----------------------------

# Single-shot detection (SSD)
Based on [this amazing tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection). 
Please refer to it for concepts and explanations. Thank you sgrvinod for an amazing work!

Below is a summary of my code structure and work flow, modified from the original tutorial. 
## Conventions: 
* class labels from 0 -> 9. Background label = 10. n_classes = 11.
* All configurations is defined in the dictionary config in main.ipynb
* When not specified, a conv layer will have `kernel_size=3, padding=1`

## Work flow: main.ipynb
1. Define Base Convolution, Auxiliary Convolution. These are better defined while seeing their structures, since the output of Base Conv is the input of Aux Conv.
2. Combine them into the SSD model
3. Create dataset class, load data
4. Train the model
5. Evaluate

## Implementation details

### Base Convolution 
```
input: 
  tensor [batch, c, h, w]
  
output: (x, fm)
  x: tensor [batch, c', h', w'] - output of the network, input to Auxiliary Convolution
  fm: list of base_conv feature maps used for detection. Each feature map is a tensor - the output of a chosen layer of the base_conv network
```  
  * Example parameter and structure: 
  ```
  conv_layers=[c1, c2, c3, c4] 
  --> module_list=[
      conv(c1, c2), norm, act_fn, 
      conv(c2, c2), norm, act_fn, 
      maxpool, 
      conv(c2, c3), norm, act_fn, 
      conv(c3, c3), norm, act_fn(*)
      maxpool,
      conv(c3, c4), norm, act_fn, 
      conv(c4, c4), norm, act_fn(*)]
   fm = [output of marked layers]
   ```
### Auxiliary Convolution
```
input: 
  tensor [batch, c', h', w'] - output x of Base Conv
  
output: 
  fm: list of aux conv feature maps used for detection. 
```
  * Example parameter and structure: 
  ```
  conv_layers=[c1,c2,c3] 
  --> module_list=[
      conv(c1, c1//2, kernel_size=1, pad=0), norm, act_fn, 
      conv(c1//2, c2, stride=2), norm, act_fn (*), 
      conv(c2, c2//2), norm, act_fn, 
      conv(c2//2, c3), norm, act_fn (*)] 
  fm = [output of marked layers]
  ```
### Prediction Convolution
```
  input: fm - appended feature maps from base_conv, aux_conv output
  ouput: (loc_output, cla_output)
    loc_output: tensor [batch, n_priors, 4]
    cla_output: tensor [batch, n_priors, n_classes]. **IMPORTANT**: raw output, NOT probability.
```
![Untitled 1](https://user-images.githubusercontent.com/43468452/131826370-d83c789e-e1af-4f5f-9238-4b7c6bfaaf3a.jpg)
The location mappings of the priors in each feature maps are then concatenated together into `loc_output`. Same goes for `cla_output`.
* SSD model 
  * Init `priors_cxcy`: tensor `[n_priors, 4]`. Fractional coordinates of all priors of this model
  * Feed forward:
  ```
  input: tensor [batch, c, h, w] - images
  ouput: (loc_output, cla_output, fm)
    loc_output: tensor [batch, n_priors, 4]
    cla_output: tensor [batch, n_priors, n_classes]. **IMPORTANT**: raw output, NOT probability.
    fm: list of feature maps used for prediction.
  ----------------------------  
  x, fm1 = base_conv(input) 
  fm2 = aux_conv(x)
  fm = fm1 + fm2 #append lists
  loc_output, cla_output = pred_conv(fm)
  return loc_output, cla_output, fm
  ```
### Detect object:
  ```
  input: 
    loc_output: tensor [batch, n_priors, 4]
    cla_output: tensor [batch, n_priors, n_classes]
    min_score, max_overlap, top_k
  output:
    all_images_boxes: list [batch] of tensor [n_boxes, 4]. n_boxes differs for each element in list
    all_images_labels: list [batch] of tensor [n_boxes]
    all_images_scores: list [batch] of tensor [n_boxes]
    
  ---------------------------
  softmax(cla_output)
  for each sample:
    for each non-background class:
      find priors who transformed into boxes has score > min score
      #Note: each prior corresponds to one box, via loc_output
      sort them decreasing score
      for each candidate box:
        unselect box with lower scores, whose overlaps with this box > max_overlap #Non-maximum Suppression
  return all_images_boxes, all_images_labels, all_images_scores
  ``` 
  
### MultiBoxLoss
```
input:     
  loc_output: tensor [batch, n_priors, 4]
  cla_output: tensor [batch, n_priors, n_classes] 
  boxes: list [batch] of tensors [n_boxes, 4], in frac. coord
  labels: list [batch] of tensors [n_boxes]
------------------------
for each image:
  Generate groundtruths loc_gt, cla_gt:
    assign each prior with the box/label with highest overlap
    assign each box/label with the prior with highest overlap
    (These priors corresponds to these boxes/labels, overwriting the above)
    loc_gt = (box coordinates) to (gcxgcy) for each prior
    priors whose overlap with their box < threshold is background
    cla_gt = class for each prior

#no more images, only (batch * n_priors) priors. 
#Reshape from [batch, n_priors, (4 or n_classes)] to [batch * n_priors, (4 or n_classes)]
positives = non-background priors
loc_loss = L1Loss(loc_output[positives], loc_gt[positives])
cla_loss = CrossEntropyLoss(cla_output, cla_gt) 
cla_loss = cla_loss[positives] + (some highest values of cla_loss[~positives])
return loc_loss + cla_loss
  ```
### Dataset class: 
  * load train.pkl
  * get the i-th item, change image to tensor, boxes and labels to list of tensors
  * images are standardized to mean=0.5, std=1 (completely empirical values)
  * define custome collate function for torch dataloader because of those list of tensors
  
### Evaluation:
```
collect all detections for all images in testset, stored in:
  all_boxes_output: list 
calculate APs, mAP between all_boxes_output and all_boxes_gt - the groundtruth detections 
```

## Experiences learnt
* In this case it is better to reduce learning rate (lr) by a small amount (`gamma = 0.5`) every few (5-10) epochs, rather than the standard `gamma = 0.1` with big (50-100) epoch steps
![train_loss](https://user-images.githubusercontent.com/43468452/131781762-e7e28a85-8030-4662-8c82-9d24161f1d86.png)

*Base model [Epoch vs train loss] plot without reducing learning rate. Note that SSD models are normally trained without reducing lr in the first 100 epochs.* `mAP = 0.79`

![train_loss (1)](https://user-images.githubusercontent.com/43468452/131782058-fac3e3b1-8e8f-4d0f-a236-3d1becf102c5.png)

*Improved model [Epoch vs train loss] plot with finer lr reduction steps, and InstanceNorm introduced into Prediction Conv.* `mAP = 0.88`

Test detections of this model at `min_score=0.4, max_overlap=0.45, top_k=200`:
![image](https://user-images.githubusercontent.com/43468452/131783034-acfd988f-fdaa-4a5a-aae5-645ca3304a60.png)

We can see that it fails to detect number 1. The Average Precision for class "1" is `0.70`.

* InstanceNorm/BatchNorm before Prediction Convolution may help, and more stable than channel-wise rescale factors 
* Remember to define priors aspect ratio/size such that they can match ground truth boxes more easily (detecting "1")
* **Try overfitting the model to one batch of data first, to make sure the model and the learning is correct!**





  
  
  

