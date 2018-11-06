# DL-Seq2Seq
This repository contains implementation of research papers on sequence-to-sequence learning. Currently the following implementations are supported:
- Machine translation (following [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) paper)
- Handwriting synthesis (following [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf) paper)
- Scheduled Sampling (following [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/pdf/1506.03099.pdf) paper)
#### I will use pre-trained models (provided in the saved_models folder) for the following demonstrations. I have also included main.py script for training the models from scratch.

## Neural Machine Translation
For this task, I have followed attentional encoder-decoder model as described in [Luong's paper](https://arxiv.org/pdf/1508.04025.pdf). I have specifically focused on **content-based attention** strategy.
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/nmt_attn.JPG" width="750">

### Train the models
If you want to train the model from scratch, then use the following command. You can set the hyperparamters in the main.py script. The trained model would be saved in **saved_model** folder.
```python    
$ python main.py
```

### Let's make some inference
For inference I have provided trained models in the **saved_model** folder. For handwriting synthesis, the pretrained models are included in the github repository, but for machine translation please download the files from [download pre-trained models for machine tranlation](https://drive.google.com/open?id=1gCqYu4UisEKgIF7R4hb9vqJXdmhA4Va2). Keep the downloaded **saved_model** folder inside the **neural machine translation** folder. You can also train your own model and parameters will be saved.

```python
>>> from eval_nmt import *

>>> encoder_e2f, decoder_e2f = load_pre_trained('eng-fra') # 'eng-fra' or 'fra-eng'
>>> encoder_f2e, encoder_f2e = load_pre_trained('fra-eng') # 'eng-fra' or 'fra-eng'

>>> eng_text = "i m not giving you any money ."
>>> fra_text = " tu es plus grande que moi ."

>>> inp1, out1, attn1 = evalText(eng_text, encoder_e2f, decoder_e2f)
English Text - "i m not giving you any money ."
French o/p   - "je ne te donnerai pas argent . <EOS>"

>>> inp2, out2, attn2 = evalText(fra_text, encoder_f2e, encoder_f2e, inp_lang='French', out_lang='English')
French Text  - "je crains de vous avoir offense ."
English o/p  - "i m afraid i ve offended you . <EOS>"

>>> viz_attn(inp1 ,out1 ,attn1)
>>> viz_attn(inp2 ,out2 ,attn2)
``` 
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/nmt.JPG" width="850">

## Handwriting Synthesis
The handwriting genration problem comes under the category of inverse problems, where we have multiple outputs at a given time-step. The idea, as given in [Alex grave's paper](https://arxiv.org/pdf/1308.0850.pdf), is to use [Mixture Density Network](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.5685&rep=rep1&type=pdf) (a gaussian distribution model) over the top of recurrent models. The handwriting generation problem is divided into two categories - unconditional and conditional generation. In case of unconditional generation the recurrent model is used to draw samples while in case of conditional generation handwriting is synthesied given some text. 

### Unconditional Generation
The unconditional model uses skip-connection as shown by arrows from input to outer recurrent model. The MDN parameters are computed by passing the output of recurrent model to MDN layer.
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/uncon.JPG" width="850">

#### Train the models
If you want to train the model from scratch, then use the following command. You can set the hyperparamters in the **main_uncond.py** script. The trained model would be saved in **saved_model** folder.
```python
$ python main_uncond.py
```

#### Let's make some inference
For inference I have provided trained models in the **saved_model** folder. If you have trained your own model, then it would overwrite the pre-trained model and can be found inside **saved_model** folder.

```python
>>> from eval_hand import load_pretrained_uncond, gauss_params_plot, plot_stroke
>>> from model import model_uncond, mdn_loss, sample_uncond, scheduled_sample

>>> lr_model, h_size = load_pretrained_uncond()
>>> strokes, mix_params = sample_uncond(lr_model, h_size)
>>> plot_stroke(strokes)
>>> gauss_params_plot(mix_params)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/unc1.JPG" width="850">

```python
>>> strokes, mix_params = sample_uncond(lr_model, h_size)
>>> plot_stroke(strokes)
>>> gauss_params_plot(mix_params)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/unc2.JPG" width="850">

### Conditional Generation
In case of handwriting synthesis, a **location based attention** mechanism is used where a attention window (w<sub>t</sub>) is convolved with the character encodings. The attention parameters k<sub>t</sub> control the location of the window, the β<sub>t</sub> parameters control the width of the window and the α<sub>t</sub> parameters control the importance of the window within the mixture.

<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/cond.JPG" width="850">

#### Train the models
If you want to train the model from scratch, then use the following command. You can set the hyperparamters in the **main_congen.py** script. The trained model would be saved in **saved_model** folder.
```python
$ python main_congen.py
```

#### Let's make some inference
For inference I have provided trained models in the **saved_model** folder. If you have trained your own model, then it would overwrite the pre-trained model and can be found inside **saved_model** folder.

```python
>>> from eval_hand import load_pretrained_congen, gauss_params_plot, plot_stroke
>>> from model import model_congen, mdn_loss, sample_congen

>>> lr_model, char_to_vec, h_size = load_pretrained_congen()
>>> strokes, mix_params, phi, win = sample_congen(lr_model, 'kiki do you love me ?', char_to_vec, h_size)
>>> plot_stroke(strokes)
>>> gauss_params_plot(mix_params)
>>> phi_window_plots(phi, win) 
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/kiki.JPG" width="850">

```python
>>> strokes, mix_params, phi, win = sample_congen(lr_model, 'a thing of beauty is joy forever', char_to_vec, h_size)
>>> plot_stroke(strokes)
>>> gauss_params_plot(mix_params)
>>> phi_window_plots(phi, win) 
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/figs/beauty.JPG" width="850">
