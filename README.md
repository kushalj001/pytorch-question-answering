# PyTorch Question Answering
This repository contains implementations of some of the most important papers for Question Answering. The implementations are in the form of tutorials and are roughly *annotations* of the said papers. This repository might be helpful for those who know the basics of deep learning and NLP, want to get started with reading slightly complex papers and see how they are implemented. This repository also assumes some familiarity with PyTorch basics, although I have tried my best to break everything down in simple terms. 
### Question Answering
Question answering is an important task based on which intelligence of NLP systems and AI in general can be judged. A QA system is given a short paragraph or *context* about some topic and is asked some questions based on the passage. The answers to these questions are spans of the context, that is they are directly available in the passage. To train such models, we use the [SQUAD](https://arxiv.org/abs/1606.05250) dataset.

## Getting Started
The notebook named __"NLP Preprocessing Pipeline for QA"__ contains all the preprocessing code that I have written. The preprocessing code does not use any high level library and I have written all the functions from scratch. It only uses spacy for tokenization. The functions implemented here are common to many NLP tasks and hence might be useful for someone who's just starting out. For example: creating vocabularies, weight matrices for pretrained embeddings, datasets/dataloaders etc. 
In hindsight, using some high-level library like torchtext would have been a better idea and I am currently working on the same.

#### Tensor Based Approach
All the notebooks are based on this approach. Ultimately, building neural nets is all about working with tensors. Knowing the shape and contents of each tensor is something that I have found very useful while learning. Hence, after each line of code, I have commented the tensor shape and changes that happen due to the transformations in code. This makes the process of understanding what's going on in neural nets more intuitive.

## Training Environment
I do not have an unlimited access to faster GPUs. The models below have been trained by renting GPUs on [vast.ai](https://vast.ai/). I used GTX 1080 Ti for majority of my experiments. 

## Papers
### 1. [DrQA](https://arxiv.org/abs/1704.00051) 
In the first notebook, we implement a comparatively simple model that involves multi-layer LSTMs and bilinear attention. The details  and intuition of each layer/component are explained before jumping into the code. This model is very similar to the one discussed in [this](https://arxiv.org/pdf/1606.02858v2.pdf) paper and also have the same first authors: [Danqi Chen](https://www.cs.princeton.edu/~danqic/). The second model is also known as the "Stanfor Attentive Reader". The model implemented in the notebook is slightly advanced version of this. The results on dev set obtained are:
Epochs | EM | F1
---|---|---|
5 | 56.4| 68.2|

I'll be training this more to improve the results and will update it soon.

### 2. [Bidirectional Attention Flow (BiDAF)](https://arxiv.org/abs/1611.01603)
Next, we move onto a bit more complex paper. This paper improves the results obtained by the previous paper. The model implemented here unlike previous ones is a multi-stage hierarchical architecture that represents the *context* and *query* at multiple levels of granularity. This paper also involves recurrence as it extensively uses LSTMs and a *memory-less* attention mechanism which is bi-directional in nature. This notebook discusses in detail about some important NLP techniques like __character embeddings__, __highway networks__ among others. Results on dev set:
Epochs | EM | F1
---|---|---|
5 | 60.4| 70.1|
 
### 3. [QANet](https://arxiv.org/abs/1804.09541)
Finally, we do away from recurrence and only use self-attention and convolutions. This paper draws inspiration from "Attention Is All You Need". The key motivation behind the design of the model is: convolution captures the local structure of the text, while the self-attention learns the global interaction between each pair of words. This tutorial explains topics like __self-attention__, and __depthwise separable convolutions__. Results on dev set:
Epochs | EM | F1
---|---|---|
3 | * | 36.6 |

I am currently training this model. I am currently short on time and I do not have access to faster GPUs. Training this for 1 epoch takes around 1 hour on GTX 1080 Ti.

## Contributions
I am not an expert. My main motive behind this project was to learn about an NLP domain. If you find any conceptual or silly mistake, kindly create an issue and I'll try my best to rectify them quickly. Other contributions are also welcome. If you train any model and get improved results, please make a PR. If you are interested in implementing more papers in this domain and wish to add them to this repository, I'd be happy to help. Although I am currently short on time, I'll be actively maintaining this repository.


