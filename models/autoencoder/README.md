# Autoencoder

This is the implementation of the encoder/decoder blocks used in (latent diffusion)[https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py] which uses convolutions and attention. 

Trained on CIFAR10

Steps             |  Reconstructions
:-------------------------:|:-------------------------:
<img src="../../images/autoencoder_tensorboard.png"/> |  <img src="../../images/autoencoder_samples.png" />

## Train

`TODO - device replication for parallel training`

```bash
pip install -r requirements.txt
```

```bash
python train_autoencoder.py
```