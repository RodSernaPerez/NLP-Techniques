# NLPutils
Description of NLP tools

## Neural Networks

### RNN

### LSTM

### Attention

- Original paper: https://arxiv.org/pdf/1512.08756.pdf
- Implementation as a keras layer in this project, obtained from: https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043#L51

### Transformer

Architecture similar to Sequence-to-Sequence that uses attention. Based on encoding-decoding.

It gives better results than RNN and CNN networks, but could have problems to deal with long dependences.

- Original paper: https://arxiv.org/pdf/1706.03762.pdf
- Official site: https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html
- Tutorial: https://medium.com/@adityathiruvengadam/transformer-architecture-attention-is-all-you-need-aeccd9f50d09
- Keras implementation: https://github.com/CyberZHG/keras-transformer

### Transformer XL


## Word Embeddings

### Word2Vec 

### Glove

### FastText 

### ELMo
ELMo is a deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy).  It is character-based.

- Official site: https://allennlp.org/elmo 

- Pre-trained models: https://github.com/HIT-SCIR/ELMoForManyLangs

- Tutorial: https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a


## Language models

### BERT

Pre-trained models for knowledge transfering that can be used with any purporse with only adding new layers. 

- Official site: https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html

- Pre-trained models: https://github.com/google-research/bert

- Tutorial: https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a
