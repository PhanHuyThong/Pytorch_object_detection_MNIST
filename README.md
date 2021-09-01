# Pytorch_tutorial_object_detection_MNIST
A newbie-friendly playground to understand and experiment object-detection in Pytorch

Training object-detection models on standard datasets can be quite computationally intensive. Here, I generate an object-detection dataset with MNIST to help learn and experiment more on the topic. This work is intended for those who want to try object detection **with no computation resources**. It will be a summary and compilation of the tutorial(s) on github I went through, as well as other algorithms I want to try implementing in the future. The content will focus on making "a summary I wished there was when I was learning it".

For the purpose of easy experimentation and establishing a work flow, the main code is presented in Colab notebooks, with core sections (models definition, training, ...) defined explicitly inside.

Let's dive right in!

# Data generation
Inspired by my friend's [work](https://github.com/nguyenvantui/mnist-object-detection/blob/master/mnist_gen.py).
See data_generation.ipynb

## Code structure
1. Load MNIST
2. Generate object-detection data:

  Set image size h,w. I recommend h = w = 32 pixels for fast experiments later on. 
  
  **I assume h = w and refer to image_size = h from this point onwards**.
  
  n_generate = number of images to generate
  
  In generating each image:
  
   - Loop through some regions of the image and randomly put a *slightly augmented* MNIST digit into that region
    
   - Record bounding boxes and labels
    
3. Save the generated datasets (see /data/train.pkl)

----------------------------
/data/train.pkl (and same for test.pkl) is a tuple of (img, boxes, labels), where

  * img: list [n_generated] of numpy arrays [h,w], dtype=int, between [0, 255].  
  
  * boxes: list [n_generated] of lists [n_objects, 4], dtype=int, between [0, h]. Boxes in boundary coordinates: (x_min, y_min, x_max, y_max). Note that n_objects can vary for each image.
  
  * labels: list [n_generated] of lists [n_objects], dtype=int. The label for each box. 
  
----------------------------

# 1. Single-shot detection (SSD)
Based on [this amazing tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection). 
**Important**: Please refer to it for concepts and explanations. Thank you sgrvinod for an amazing work!

Below is a summary of my code structure and work flow, modified from the original tutorial. 
## Conventions: 
* class labels from 0 -> 9. Background label = 10. n_classes = 11.
* All configurations is defined in the dictionary config in main.ipynb
* When not specified, a conv layer will have `kernel_size=3, padding=1`

## Work flow: main.ipynb
1. Define Base Convolution, Auxiliary Convolution. These are better defined while seeing their structures
2. Combine them into the SSD model
3. Create dataset class, load data
4. Train the model
5. Evaluate

## Implementation details

* Base Convolution 
  * `input`: tensor `[batch, channel, h, w]`
  * `output`: `(x, fm)` 
  
  `x`: output of the network. This is input to Auxiliary Convolution.
  
  `fm`: list of base conv feature maps used for detection. This automatically includes the last 2 feature maps. Note `x = fm[-1]` here, but can be modified.
  
  * Example parameter and structure: 
  ```
    conv_layers=[1,10,20] 
    --> module_list=[
        conv(1,10), norm, act_fn, 
        conv(10,10), norm, act_fn, 
        maxpool, 
        conv(10, 20), norm, act_fn, 
        conv(20, 20), norm, act_fn]
   ```
* Auxiliary Convolution
  * `input`: tensor, the last feature map of Base Convolution (default). Could be any feature map though.
  * `output`: list of aux conv feature maps used for detection. This automatically includes every feature maps similar to [1](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)
  * Example parameter and structure: 
  ```
  conv_layers=[c1,c2,c3] 
  --> module_list=[
      conv(c1, c1//2, kernel_size=1, pad=0), norm, act_fn, 
      conv(c1//2, c2, 3, 1, stride=2), norm, act_fn, 
      conv(c2, c2//2), norm, act_fn, 
      conv(c2//2, c3), norm, act_fn]
   ```
* Prediction Convolution
  * `input`: list of all feature maps for detection, collected from `aux_conv` and `base_conv`
  * `ouput`: 
    * `loc_output`: tensor `[batch, n_priors, 4]` 
    * `cla_output`: tensor `[batch, n_priors, n_classes]`. **IMPORTANT**: raw output, NOT probability.
* SSD model 
  * `priors_cxcy`: tensor `[n_priors, 4]`. Fractional coordinates of all priors of this model
  * Feed forward:
  ```
  input [batch, c, h, w]

  x, fm1 = base_conv(input) 
  fm2 = aux_conv(x)
  fm = fm1 + fm2 #append lists
  loc_output, cla_output = pred_conv(fm)
  
  return loc_output, cla_output, fm
  ```
  * Detect object:
  ```
  input: loc_output [batch, n_priors, 4], cla_output [batch, n_priors, n_classes], min_score, max_overlap, top_k
  
  softmax(cla_output)
  for each sample:
    for each non-background class:
      find priors with score > min score
      sort them decreasing score
      for each candidate prior:
        unselect priors with lower scores, whose overlaps with this guy > max_overlap
  collect all selected priors
  return 
  
* MultiBoxLoss
  * `input`:  `loc_output, cla_output, boxes, labels`
  * Code:
  ```
  for each image:
    assign each prior with the box/label with highest overlap
    assign each box/label with the prior with highest overlap. Those priors corresponds to those boxes/labels, overwriting the above.
    loc_gt = (box coordinates) to (gcxgcy) for each prior
    priors whose overlap with their box < threshold is background
    cla_gt = class for each prior
  
  #no more images, only `(batch * n_priors)` priors. Reshaping is important here. See code for details.
  positives = non-background priors
  loc_loss = L1Loss(loc_output[positives], loc_gt[positives])
  cla_loss = CrossEntropyLoss(cla_output, cla_gt) 
  cla_loss = cla_loss[positives] + (some highest values of cla_loss[~positives])
  return loc_loss + cla_loss
  ```


  
  
  

