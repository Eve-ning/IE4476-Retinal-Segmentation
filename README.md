# Retinal Segmentation

This project is part of IE4476 Image Processing, a course offered by NTU

## Running it

You need to create a Python 3.8 environment, then:

```
pip install -r requirements.txt
python main.py
```

## Summary

We analyzed and fine-tuned a single train image to segment a vessel boolean image.

| Train Image Input      | Train Image Expected Output | Train Image Actual Output  |
| ---------------------- | --------------------------- | -------------------------- |
| ![](data/x_train.gif)  | ![](data/y_train.gif)       | ![](data/y_pred_train.gif) |

### Results

Take results with caution, we only used 1 training sample and tested with that sample.

| F1 Score | Accuracy | Sensitivity (Recall) |
|----------|----------|----------------------|
| 80.06%   | 78.75%   | 95.30%               |

## Other Tests

We also ran it against images without ground truths

![](data/other_tests.png)
