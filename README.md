# Video Anomaly Detection - powered by Azure MLOps
[![Build Status](https://dev.azure.com/aidemos/MLOps/_apis/build/status/Microsoft.MLOps_VideoAnomalyDetection?branchName=master)](https://dev.azure.com/aidemos/MLOps/_build/latest?definitionId=88?branchName=master)

The automation of detecting anomalous events in videos is a challenging problem that currently attracts a lot of attention by researchers, but also has broad applications across industry verticals.  

The approach involves training deep neural networks to develop an in-depth understanding of the physical and causal rules in the observed scenes. The model effectively learns to predict future frames in the video in a self-supervised fashion. 

By calculating the error in this prediction, it is then possible to detect if something unusual, an anomalous event, occurred, if there is a large prediction error.  

The approach can be used both in a supervised and unsupervised fashion, thus enabling the detection of pre-defined anomalies, but also of anomalous events that have never occurred in the past. 

> Post on [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:6512538611181846528) (includes video demonstration)

# Learning Goals

You will learn:
1. How to adapt an existing neural network architecture to your use-case.
1. How to prepare video data for deep learning. 
1. How to perform hyperparameter tuning with [HyperDrive](https://azure.microsoft.com/en-us/blog/experimentation-using-azure-machine-learning/) to improve the performance of you model.
1. How to deploy a deep neural network as a webservice for video processing. 
1. How to post-process the output of a Keras model for secondary tasks (here, anomaly detection)
2. How to define a build pipeline for DevOps.


# Pre-requisites

## Skills

1. Some familiarity with concepts and frameworks for neural networks:
	- Framework: [Keras](https://keras.io/)
	- Concepts: [convolutional](https://keras.io/layers/convolutional/), [recurrent](https://keras.io/layers/recurrent/), and [pooling](https://keras.io/layers/pooling/) layers.
2. Knowledge of basic data science and machine learning concepts. [Here](https://www.youtube.com/watch?v=gNV9EqwXCpw) and [here](https://www.youtube.com/watch?v=GBDSBInvz08) you'll find short introductory material.
3. Moderate skills in coding with Python and machine learning using Python. A good place to start is [here](https://www.youtube.com/watch?v=-Rf4fZDQ0yw&list=PLjgj6kdf_snaw8QnlhK5f3DzFDFKDU5f4).

## Software Dependencies

- Various python modules. We recommend working with a conda environement (see `environment.yml`) - [Documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- VS code [https://code.visualstudio.com/](https://code.visualstudio.com/)
- X2Go [https://wiki.x2go.org/doku.php](https://wiki.x2go.org/doku.php)

We found that a useful development environment is to have a VM with a GPU and connect to it using X2Go.

## Hardware Dependencies

A computer with a GPU, Standard NC6 sufficient, faster learning with NC6_v2/3 or ND6. [compare VM sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu)

## Dataset

[UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)

## Agenda

### Getting Started

1. [Data Preparation](./docs/data_prep_w_pillow.md)
2. [Model Development](./docs/model_development.md)
3. [Hyperparameter Tuning](./docs/hyperparameter_tuning.md)
4. [Anomaly Detection](./docs/anomaly_detection.md)
5. [Deployment](./docs/deployment.md)

### Advanced Topics

1. Transfer learning - (How to quickly train the model on different source of video)
1. MLOps - See this [blog post](https://azure.microsoft.com/en-us/blog/breaking-the-wall-between-data-scientists-and-app-developers-with-azure-devops/)

## References / Resources

- Research Article: [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104) by Lotter, W., Kreiman, G. and Cox, D., 2016.

	```
	@article{lotter2016deep,
	title={Deep predictive coding networks for video prediction and unsupervised learning},
	author={Lotter, William and Kreiman, Gabriel and Cox, David},
	journal={arXiv preprint arXiv:1605.08104},
	year={2016}
	}
	```
- Original Prednet implentation is on [github.com](https://coxlab.github.io/prednet/). Note, that the original implementation will only work in Python 2, but not in Python 3.

- Interesting blog post on [Self-Supervised Video Anomaly Detection](https://launchpad.ai/blog/video-anomaly-detection) by [Steve Shimozaki](https://launchpad.ai/blog?author=590f381c3e00bed4273e304b) 
