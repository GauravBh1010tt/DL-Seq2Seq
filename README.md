# DL-Seq2Seq
This repository contains implementation of research papers on sequence-to-sequence learning. Currently the following implementations are supported:
- Machine translation (following [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) paper)
- Handwriting synthesis (following [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf) paper)
## Neural Machine Translation
```python
>>> text = 'hello how are you'
>>> evaluate(text)
French Text  - "il est tres sociable ."
Ground Truth - "he is very sociable ."
English o/p  - "he is very sociable . <EOS>"
```

## Handwriting Synthesis
### Unconditional Generation
```python
>>> text = 'hello how are you'
>>> evaluate(text)
```
