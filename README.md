# NLPutils
Description of NLP tools

## Neural Networks

### Recurrent Neural Network (RNN)

Neural networks to deal with temporal variables.

Has problems to deal with long dependences.

- [Tutorial](https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/)
- Implemented in [Keras.layers](https://keras.io/layers/recurrent/) .

### LSTM

Improvement of RNN that better deals with long dependecnes.

- [Original paper](http://www.bioinf.jku.at/publications/older/2604.pdf)
- Implemented in [Keras.layers](https://keras.io/layers/recurrent/) .

### Attention

- [Original paper](https://arxiv.org/pdf/1512.08756.pdf)
- Implementation as a keras layer in this project, obtained from [here](https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043#L51)

### Transformer

Architecture similar to Sequence-to-Sequence that uses attention. Based on encoding-decoding.

It gives better results than RNN and CNN networks, but could have problems to deal with long dependences.

- [Original paper](https://arxiv.org/pdf/1706.03762.pdf)
- [Official site](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- [Tutorial](https://medium.com/@adityathiruvengadam/transformer-architecture-attention-is-all-you-need-aeccd9f50d09)
- [Keras implementation](https://github.com/CyberZHG/keras-transformer)

### Transformer XL

Combination of RNNs and Transformer to deal with the problem of long dependences. Current state of the art.
- [Original paper](https://arxiv.org/abs/1901.02860)
- [Official site](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html)
- [Tutorial](https://towardsdatascience.com/transformer-xl-explained-combining-transformers-and-rnns-into-a-state-of-the-art-language-model-c0cfe9e5a924)
- Tensorflow & PyTorch implementations [here](https://github.com/kimiyoung/transformer-xl)

## Word Embeddings

Deep vector representations of words. 
They can be contextual or character based:
- Contextual: if the information of the surrounding words is used. Can be CBOW architecture if predicts the current word based on the
context or Skip-gram if predicts surrounding words given the current word.
- Character: if the vectors are computed based on the characters that form the word.


### Word2Vec 

- [Original paper](https://arxiv.org/pdf/1301.3781.pdf)
- [Official site](https://code.google.com/archive/p/word2vec/)
- [English pretained model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
- Very easy to compute and manage with [Gensim library](https://radimrehurek.com/gensim/) for python. 

### Glove

GloVe is modeled to do dimensionality reduction in the co-occurrence counts matrix.

- [Official site and pretrained models](https://nlp.stanford.edu/projects/glove/)

### FastText 

Vector generation based on characters of the word. 

- [Tutorial](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3)
- Very easy to compute and manage with [Gensim library](https://radimrehurek.com/gensim/) for python. 

### ELMo
ELMo is a deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy).  It is character-based.

- [Official site](https://allennlp.org/elmo)

- [Pre-trained models](https://github.com/HIT-SCIR/ELMoForManyLangs)

- [Tutorial](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)


## Language models

### BERT

Pre-trained models for knowledge transfering that can be used with any purporse with only adding new layers. 

- [Original paper](https://arxiv.org/pdf/1810.04805.pdf)

- [Official site](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

- [Pre-trained models](https://github.com/google-research/bert)

- [Tutorial](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
