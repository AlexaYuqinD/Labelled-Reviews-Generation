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
<img src="https://github.com/AlexaYuqinD/Labelled-Reviews-Generation/blob/master/images/VAE.PNG" 
 width="400" height="110" />
</p>

## 4. Experiments

### Pitchfork


### Sentiment TB


### Amazon Reviews



## 5. Error Analysis

## Reference

[rohithreddy024/VAE-Text-Generation](https://github.com/rohithreddy024/VAE-Text-Generation)

[Denny Britz, Anna Goldie, Thang Luong, and Quoc Le.](https://arxiv.org/abs/1703.03906)

[Irina Higgins, Loic Matthey, Arka Pal, et al.](https://openreview.net/forum?id=Sy2fzU9gl)

[Matthew Honnibal and Ines Montani.](https://github.com/explosion/spaCy)

[Kihyuk Sohn, Honglak Lee, and Xinchen Yan.](https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf)