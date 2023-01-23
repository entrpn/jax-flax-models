import jax
import jax.numpy as jnp

from autoencoder import ResnetBlock, AttnBlock, DownSample, Encoder, Upsample, Decoder, Autoencoder

main_rng = jax.random.PRNGKey(42)
print("Device:",jax.devices()[0])

main_rng, x_rng = jax.random.split(main_rng)
x = jax.random.normal(x_rng,(3,32,32,64))

res_block = ResnetBlock(in_channels=64, out_channels=32)

main_rng, init_rng, dropout_init_rng = jax.random.split(main_rng, 3)

params = res_block.init({'params':init_rng, 'dropout':dropout_init_rng},x,train=True)['params']

main_rng, dropout_rng = jax.random.split(main_rng)
out = res_block.apply({'params':params},x,train=True, rngs={'dropout': dropout_rng})
print("resnet out",out.shape)

main_rng, init_rng, dropout_init_rng = jax.random.split(main_rng, 3)

attn_block = AttnBlock(64,1)
params = attn_block.init({'params' : init_rng}, x, train=True)['params']

main_rng, dropout_rng = jax.random.split(main_rng)

out = attn_block.apply({
    'params' : params}, x, train=True,
    rngs={'dropout' : dropout_rng})

print("attn out",out.shape)

main_rng, init_rng, dropout_init_rng = jax.random.split(main_rng, 3)
downsample = DownSample(64,True)
params = downsample.init({'params' : init_rng}, x)['params']
main_rng, dropout_rng = jax.random.split(main_rng)
out = downsample.apply({
    'params' : params}, x)

print("downsample out",out.shape)

main_rng, init_rng, dropout_init_rng = jax.random.split(main_rng, 3)
upsample = Upsample(64,True)
params = upsample.init({'params' : init_rng}, x)['params']
main_rng, dropout_rng = jax.random.split(main_rng)
out = upsample.apply({
    'params' : params}, x)

print("upsample out",out.shape)

main_rng, x_rng = jax.random.split(main_rng)
x = jax.random.normal(x_rng,(3,32,32,3))
main_rng, init_rng, dropout_init_rng = jax.random.split(main_rng, 3)
encoder = Encoder(ch=64, ch_mult=[1,2,3,4], num_res_blocks=2, attn_resolutions=[64, 32],resolution=64, z_channels=512)
params = encoder.init({'params' : init_rng}, x)['params']
main_rng, dropout_rng = jax.random.split(main_rng)
out = encoder.apply({
    'params' : params}, x)

print("encoder out",out.shape)

main_rng, init_rng, dropout_init_rng = jax.random.split(main_rng, 3)
decoder = Decoder(ch=64, out_ch=3, ch_mult=[1,2,3,4], num_res_blocks=2, attn_resolutions=[64, 32],resolution=64, z_channels=512)
params = decoder.init({'params' : init_rng}, out)['params']
main_rng, dropout_rng = jax.random.split(main_rng)
out = decoder.apply({
    'params' : params}, out)

print("decoder out",out.shape)

main_rng, x_rng = jax.random.split(main_rng)
x = jax.random.normal(x_rng,(10,32,32,3))
main_rng, init_rng, dropout_init_rng = jax.random.split(main_rng, 3)
autoencoder = Autoencoder(ch=32, out_ch=3, ch_mult=[1,2,3,4], num_res_blocks=2, attn_resolutions=[64, 32],resolution=32, z_channels=512)
params = autoencoder.init({'params' : init_rng}, x)['params']
main_rng, dropout_rng = jax.random.split(main_rng)
out = autoencoder.apply({
    'params' : params}, x)

print("autoencoder out",out.shape)