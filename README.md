# CNNImageRetrieval: Training and evaluating CNNs for Image Retrieval

**CNNImageRetrieval** is a MATLAB toolbox that implements the training and testing of the approach described in our ECCV 2016 paper "CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples" [[arXiv](http://arxiv.org/abs/1604.02426)]

<img src="http://cmp.felk.cvut.cz/cnnimageretrieval/cnnimageretrieval_teaser.png" width=\textwidth/>

## What is it?

This code implements:

1. Training (fine-tuning) CNN for image retrieval
2. Testing CNN image retrieval on Oxford5k and Paris6k datasets

## Prerequisites

In order to run this toolbox you will need:

1. MATLAB (tested with MATLAB R2017a on Debian 8.1)
2. MatConvNet MATLAB toolbox version 1.0-beta24 [[matconvnet-1.0-beta24.tar.gz](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta24.tar.gz)]
3. All the rest (data + networks) is automatically downloaded with our scripts

## Execution

Run the following script in MATLAB:

```
>> run [MATCONVNET_ROOT]/matlab/vl_setupnn;
>> run [CNNIMAGERETRIEVAL_ROOT]/setup_cnnimageretrieval;
>> train_cnnimageretrieval;
>> test_cnnimageretrieval;
```

## Citation

If you use this work please cite our ECCV 2016 publication "CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples" [[arXiv](http://arxiv.org/abs/1604.02426)]

Bibtex:
```
@inproceedings{RTC16,
 title = {{CNN} Image Retrieval Learns from {BoW}: Unsupervised Fine-Tuning with Hard Examples},
 author = {Radenovi{\'c}, F. and Tolias, G. and Chum, O.},
 booktitle = {ECCV},
 year = {2016}
}
