# AgeEstimator
The repository contains inference wrapper for a code originally
placed in https://github.com/Raschka-research-group/coral-cnn.

## Weights
* Original source: [archive](https://drive.google.com/drive/folders/1Q9vr5Q0BueHD0Kal2pEmA-NWIbYZnJAL), 
[description](https://github.com/Raschka-research-group/coral-cnn/tree/master/single-image-prediction).
* Mirror: [weights](https://drive.google.com/open?id=1u8ZIHHWkXqpfp_HT-Jzr8R6MrozdDn7A).

## Implementation details
Only one model from a group proposed in https://github.com/Raschka-research-group/coral-cnn
was implemented. The chosen one was the model trained on MORPH dataset that 
was trained with use of Cross-Entropy loss.

## Installation
```bash
/path/to/repository_root$ pip install .
``` 

## API overview

### Importing
In order to import the inference wrapper one need to create the following 
import statement:
```python
from age_estimator import AgeEstimator
```

### Initialization
To initialize the **AgeEstimator** object the easiest way is to use:
```python
from age_estimator import AgeEstimator


age_estimator = AgeEstimator.initialize(
    weights_path="path_to_pre_fetched_weights",
    min_prediction_probability=0.4,
    use_gpu=False
)
```

**min_prediction_probability** states minimum confidence value of top prediction
to classify inference result as valid.

### Usage
```python
from age_estimator import AgeEstimator
import cv2 as cv

age_estimator = AgeEstimator.initialize(
    weights_path="path_to_pre_fetched_weights"
)


# image must be an image of size (H, W, 3) RGB/BRG mode - otherwise 
# ValueError will be raised.
image = cv.imread("path_to_video")
result = age_estimator.estimate_age(
    image=image,
    image_in_rgb_mode=False # adjust if input image transformed in runtime!
)

```
 