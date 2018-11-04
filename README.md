# DL-Seq2Seq
This repository contains implementation of research papers on sequence-to-sequence learning. Currently the following implementations are supported:
- Machine translation (following [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) paper)
- Handwriting synthesis (following [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf) paper)
- Scheduled Sampling (following [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/pdf/1506.03099.pdf) paper)
#### I will use pre-trained models (provided in the saved_models folder) for the following demonstrations. I have also included main.py script for training the models from scratch.

## Neural Machine Translation
For this task, I have followed attentional encoder-decoder model as described in [Luong's paper](https://arxiv.org/pdf/1508.04025.pdf). I have specifically focused on **content-based attention** strategy.
### Train the models
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
The handwriting genration problem comes under the category of inverse problems, where we have multiple outputs at a given time-step. The idea, as given in [Alex grave's paper](https://arxiv.org/pdf/1308.0850.pdf), is to use [Mixture Density Network](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.5685&rep=rep1&type=pdf) (a gaussian distribution model) over the top of recurrent models. The handwriting generation problem is divided into two categories - unconditional and conditional generation. In case of unconditional generation the recurrent model is used to draw samples while in case of conditional generation handwriting is synthesied given some text. 

### Unconditional Generation

<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/uncon.JPG" width="850">

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
In case of handwriting synthesis, a location based attention mechanism is used where a attention window (w<sub>t</sub>) is convolved with the character encodings. The attention parameters k<sub>t</sub> control the location of the window, the β<sub>t</sub> parameters control the width of the window and the α<sub>t</sub> parameters control the importance of the window within the mixture.

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
