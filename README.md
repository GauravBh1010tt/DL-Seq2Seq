# DL-Seq2Seq
This repository contains implementation of research papers on sequence-to-sequence learning. Currently the following implementations are supported:
- Machine translation (following [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) paper)
- Handwriting synthesis (following [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf) paper)
- Scheduled Sampling (following [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/pdf/1506.03099.pdf) paper)
## Neural Machine Translation
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/nmt_attn.JPG" width="750">

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
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/uncond.JPG" width="850">
```python
>>> strokes, mix_params = sample(lr_model, time_steps=800, random_state = 1283)
>>> plot_stroke(strokes)
>>> gauss_params_plot(mix_params)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/unc1.JPG" width="850">

```python
>>> strokes, mix_params = sample(lr_model, time_steps=800, random_state = 35442)
>>> plot_stroke(strokes)
>>> gauss_params_plot(mix_params)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/unc2.JPG" width="850">

### Conditional Generation

<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/cond.JPG" width="850">

```python
>>> strokes, mix_params, phi, win = sample(lr_model, 'kiki do you love me ?', char_to_vec)
>>> phi_window_plots(phi, win) 
>>> plot_stroke(strokes)
>>> gauss_params_plot(mix_params)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/kiki.JPG" width="850">

```python
>>> strokes, mix_params, phi, win = sample(lr_model, 'a thing of beauty is joy forever', char_to_vec)
>>> phi_window_plots(phi, win) 
>>> plot_stroke(strokes)
>>> gauss_params_plot(mix_params)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/beauty.JPG" width="850">
