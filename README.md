<div align="center">

<h2>
Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples
</h2>

<b>CVPR 2026 Workshop</b>

Reiji Saito, Satoshi Kamiya, Kauzhiro Hotta
</div>

## Install

Create a new conda environment and install required packages.
(This procedure is for GLASS and RePaste. For other methods, please refer to the original configurations and implementations.)
```
conda create -n RePaste python=3.9
conda activate RePaste
pip install -r requirements.txt
```

### Download dataset

Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip the file to `../mvtec`.

fg_mask is the same as in GLASS. Please refer to the [GLASS implementation](https://github.com/cqylunlun/glass).

### A2N setting
```
|-- mvtec
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
    |-- zipper
    |-- fg_mask
            |-- bottle
            |-- cable
            |-- capsule
            |-- ....
            |-- zipper
```
### N2A setting
We generated synthetic anomalies using [AnomalyAny](https://github.com/EPFL-IMOS/AnomalyAny#-news) and [MemSeg](https://github.com/TooTouch/MemSeg). If there are better methods for generating synthetic anomalies, it would be preferable to use those instead.
We generated 20 synthetic anomaly images for training and 20 for testing in each category. Similarly, we created corresponding synthetic anomaly masks, and produced ground-truth annotations for both the training and test synthetic anomalies.
```
|-- mvtec
    |-- bottle
        |-- train
            |-- good
                |-- 000.png
                |-- ....
                |-- 208.png
                |-- pseudo_0000.png
                |-- ....
                |-- pseudo_0019.png
        |-- test
                |-- broken_large
                |-- broken_small
                |-- contamination
                |-- good
                |-- pseudo_anomaly
                    |-- pseudo_0000.png
                    |-- ....
                    |-- pseudo_0019.png
        |-- ground_truth_train (Option)
                |-- pseudo_0000.png
                |-- ....
                |-- pseudo_0019.png
        |-- ground_truth
                |-- broken_large
                |-- broken_small
                |-- contamination
                |-- pseudo_anomaly
                    |-- pseudo_0000_mask.png
                    |-- ....
                    |-- pseudo_0019_mask.png
    |-- cable
    |-- capsule
    |-- ....
    |-- zipper
    |-- fg_mask
            |-- bottle
            |-- cable
            |-- capsule
            |-- ....
            |-- zipper
```
### How to run
```
# A2N
bash run/A2N.sh

# N2A
bash run/N2A.sh
```
