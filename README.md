# CNNImageRetrieval: Training and evaluating CNNs for Image Retrieval

**CNNImageRetrieval** is a MATLAB toolbox that implements the training and testing of the approach described in our papers:

> *Fine-tuning CNN Image Retrieval with No Human Annotation*, 
> Radenović F., Tolias G., Chum O., 
> arXiv 2017 [[arXiv](https://arxiv.org/abs/1711.02512)]

> *CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples*, 
> Radenović F., Tolias G., Chum O., 
> ECCV 2016 [[arXiv](http://arxiv.org/abs/1604.02426)]

<img src="http://cmp.felk.cvut.cz/cnnimageretrieval/cnnimageretrieval_teaser.png" width=\textwidth/>

## What is it?

This code implements:

1. Training (fine-tuning) CNN for image retrieval
1. Learning supervised whitening for CNN image representations
1. Testing CNN image retrieval on Oxford5k and Paris6k datasets

## Prerequisites

In order to run this toolbox you will need:

1. MATLAB (tested with MATLAB R2017a on Debian 8.1)
1. MatConvNet MATLAB toolbox version [1.0-beta25](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz)
1. All the rest (data + networks) is automatically downloaded with our scripts

## Execution

Run the following script in MATLAB:

```
>> run [MATCONVNET_ROOT]/matlab/vl_setupnn;
>> run [CNNIMAGERETRIEVAL_ROOT]/setup_cnnimageretrieval;
>> train_cnnimageretrieval;
>> test_cnnimageretrieval;
```

## Citation

Related publications:
```
@inproceedings{Radenovic-arXiv17a,
 title={Fine-tuning {CNN} Image Retrieval with No Human Annotation},
 author={Radenovi{\'c}, Filip and Tolias, Giorgos and Chum, Ond{\v{r}}ej},
 booktitle = {arXiv:1711.02512},
 year={2017}
}
```
```
@inproceedings{RTC16,
 title = {{CNN} Image Retrieval Learns from {BoW}: Unsupervised Fine-Tuning with Hard Examples},
 author = {Radenovi{\'c}, F. and Tolias, G. and Chum, O.},
 booktitle = {ECCV},
 year = {2016}
}
```