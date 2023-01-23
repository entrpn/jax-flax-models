from vision_transformer import AttentionBlock, ViT
import jax
from jax import random

# Seeding for random operations
main_rng = random.PRNGKey(42)

print("Device:", jax.devices()[0])

main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (3, 16, 128))

attnblock = AttentionBlock(embed_dim=128, hidden_dim=512, num_heads=4, dropout_prob=0.1)

main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = attnblock.init({'params' : init_rng, 'dropout' : dropout_init_rng}, x, train=True)['params']

main_rng, dropout_init_rng = random.split(main_rng)
out = attnblock.apply({'params' : params}, x, train=True, rngs={'dropout' : dropout_init_rng})

print('out',out.shape)

print("VIT TEST*********************")
print()

main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (5, 32, 32, 3))
# Create vision transformer
visntrans = ViT(embed_dim=128,
                              hidden_dim=512,
                              num_heads=4,
                              num_channels=3,
                              num_layers=6,
                              num_classes=10,
                              patch_size=4,
                              num_patches=64,
                              dropout_prob=0.1)
# Initialize parameters of the Vision Transformer with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = visntrans.init({'params': init_rng, 'dropout': dropout_init_rng}, x, True)['params']
# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
out = visntrans.apply({'params': params}, x, train=True, rngs={'dropout': dropout_apply_rng})
print('Out', out.shape)