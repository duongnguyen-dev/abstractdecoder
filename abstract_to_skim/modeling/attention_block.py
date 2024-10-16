import tensorflow as tf 
from abstract_to_skim.backbone.feed_forward_block import FeedForwardBlock

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.multihead = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()   
        self.ff_dim = ff_dim
        self.ff = FeedForwardBlock(embed_dim=embed_dim, ff_dim=ff_dim, dropout_rate=dropout_rate)
    
    def call(self, x):
        attn_output = self.multihead(
            query = x,
            value = x,
            key = x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        x = self.ff(x)

        return x