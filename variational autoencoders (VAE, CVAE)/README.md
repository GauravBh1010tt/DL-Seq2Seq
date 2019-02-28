# Variational Autoencoder (VAE)

## Usage
The script to run the codes are given in ```vae.py```. For a quick run, download all contents in a single folder and run:
```python
$ python vae.py
```
## Inference
```python
>>> sample_fig(encoder, decoder, x_train, x_test)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/zzfigs/vae1.JPG" width="400">

```python
>>> hallucinate(decoder, n_samples=30)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/zzfigs/vae2.JPG" width="520">

# Conditional Variational Autoencoder (CVAE)

## Usage
The script to run the codes are given in ```vae.py```. For a quick run, download all contents in a single folder and run:
```python
$ python cvae.py
```
## Inference
```python
>>> sample_fig(encoder, decoder, x_train, x_test)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/zzfigs/cvae.JPG" width="400">

```python
>>> hallucinate(decoder, n_samples=50)
```
<img src="https://github.com/GauravBh1010tt/DL-Seq2Seq/blob/master/zzfigs/cvae2.JPG" width="520">
