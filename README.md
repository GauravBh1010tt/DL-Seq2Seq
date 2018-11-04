# DL-Seq2Seq
This repository contains implementation of research papers on sequence-to-sequence learning. Currently the following implementations are supported:
- Machine translation (following [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) paper)
- Handwriting synthesis (following [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf) paper)
## Neural Machine Translation
```python
>>> text1 = "je crains de vous avoir offense ."
>>> text2 = " tu es plus grande que moi ."

>>> inp1, out1, attn1 = evalText(text1)
French Text  - "je crains de vous avoir offense ."
English o/p  - "i m afraid i ve offended you . <EOS>"

>>> inp2, out2, attn2 = evalText(text2)
French Text  - "tu es plus grande que moi ."
English o/p  - "you re taller than me . <EOS>"

>>> vis_attn(inp1 ,out1 ,attn1)
>>> vis_attn(inp2 ,out2 ,attn2)
``` 
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/nmt.JPG" width="850">

## Handwriting Synthesis
### Unconditional Generation
```python
>>> text = 'hello how are you'
>>> evaluate(text)
```
