# Labelled Reviews Generation

## Collaborators

[Haoshen Hong](https://github.com/HowsenH)

[Hao Li](https://github.com/haoli94)


## 1. Introduction

Nowadays, while most generative tasks focus on image generation, one of the
most challenging tasks for generative models is to simulate natural language
distribution and generate realistic sentences. This project focuses on combining a
Conditional Variational Autoencoder with a Encoder-Decoder architecture to learn
the distribution of encoded sentences and generate realistic word sequences with
control variables.

## 2. Data
- [Pitchfork](https://www.kaggle.com/bcyphers/pitchfork-reviews)
- [Sentiment TB](https://nlp.stanford.edu/sentiment/)
- [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/)


[link to our data](https://drive.google.com/open?id=1ZtNRlMHObf6EPQHKM1daGu7Y8EPUKyn2)

## 3. Model

<p align="center">Model Architecture</p>
<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/VAE.png" 
 width="720" height="200" />
</p>

## 4. Experiments

### Pitchfork
Training loss in 100,000 iterations of Pitchfork. From left to right are ELBO, KL, reconstruction error and the weighted training objective.
<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/pitch.png" 
 width="700" height="200" />
</p>

### Sentiment TB
Training and testing loss in 55,000 iterations of Sentiment TB. Up: Training. Down: Testing. From left to right are ELBO, KL, reconstruction error and the weighted training objective.
<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/sentiment.jpg" 
 width="650" height="250" />
</p>

Samples generated from encoded Gaussian of Sentiment TB dataset. Distribution of each category is approximately unit Gaussian, which corresponds to defined prior.
<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/latent.png" 
 width="650" height="270" />
</p>

### Amazon Reviews
Training and testing loss in 10,000 iterations of Amazon Reviews with hidden size 512. Up: Training. Down: Testing. From left to right are ELBO, KL, reconstruction error and the weighted
training objective.
<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/amazon_512.jpg" 
 width="600" height="250" />
</p>

Training and testing loss in 45,000 iterations of Amazon Reviews with hidden size 256. Up: Training. Down: Testing. From left to right are ELBO, KL, reconstruction error and the weighted
training objective.
<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/amazon_256.jpg" 
 width="600" height="250" />
</p>

## 5. Error Analysis

Major problems: unclear sentiment and too many <unk\> tokens.

Possible explanations: 

- The dataset is noisy.
- The size of dictionary might be too small.

### Pitchfork

<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/error1.PNG" 
 width="400" height="150" />
</p>


### Sentiment TB
<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/error2.PNG" 
 width="400" height="150" />
</p>


### Amazon Reviews
<p align="center">
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/error3.PNG" 
 width="400" height="150" />
</p>


## Reference

[rohithreddy024/VAE-Text-Generation](https://github.com/rohithreddy024/VAE-Text-Generation)

[Denny Britz, Anna Goldie, Thang Luong, and Quoc Le.](https://arxiv.org/abs/1703.03906)

[Irina Higgins, Loic Matthey, Arka Pal, et al.](https://openreview.net/forum?id=Sy2fzU9gl)

[Matthew Honnibal and Ines Montani.](https://github.com/explosion/spaCy)

[Kihyuk Sohn, Honglak Lee, and Xinchen Yan.](https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf)