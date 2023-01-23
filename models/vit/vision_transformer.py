import numpy as np
import jax.numpy as jnp
import flax.linen as nn

def img_to_patch(x, patch_size, flatten_channels=True, overlap=False):
    b,h,w,c = x.shape
    x = x.reshape(b, h//patch_size, patch_size, w//patch_size, patch_size, c)
    x = x.transpose(0,1,3,2,4,5)
    x = x.reshape(b,-1,*x.shape[3:])
    if flatten_channels:
        x = x.reshape(b,x.shape[-1],-1)
    return x

class AttentionBlock(nn.Module):
    embed_dim : int
    hidden_dim : int
    num_heads : int
    dropout_prob : float = 0.0

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim)
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)
    
    def __call__(self,x,train=True):
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + self.dropout(attn_out, deterministic= not train)

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out,deterministic=not train)
        
        x = x + self.dropout(linear_out, deterministic=not train)

        return x

class ViT(nn.Module):
    embed_dim : int
    hidden_dim : int #Dim of hidden layer in feed-forward network
    num_heads : int
    num_channels : int
    num_layers : int #Number of blocks to use in transformer
    num_classes : int
    patch_size : int
    num_patches : int
    dropout_prob : float = 0.0

    def setup(self):
        self.input_layer = nn.Dense(self.embed_dim)
        self.transformer = [AttentionBlock(self.embed_dim, self.hidden_dim, num_heads=4, dropout_prob=0.1) for _ in range(self.num_layers)]

        self.mlp_head = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.num_classes)
        ])
        self.dropout = nn.Dropout(self.dropout_prob)

        self.cls_token = self.param('cls_token',nn.initializers.normal(stddev=1.0),(1,1,self.embed_dim))

        self.pos_embedding = self.param('pos_embedding',nn.initializers.normal(stddev=1.0),(1,1+self.num_patches, self.embed_dim))
    
    def __call__(self, x, train=True):
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        cls_token = self.cls_token.repeat(B, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embedding[:,:T+1]

        x = self.dropout(x, deterministic=not train)
        for attn_block in self.transformer:
            x = attn_block(x, train=train)
        
        cls = x[:,0]
        out = self.mlp_head(cls)
        return out