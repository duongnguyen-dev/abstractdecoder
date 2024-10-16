import tensorflow as tf

class FeedForwardBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layernorm(x)

        return x
    
# if __name__ == "__main__":
#     sample = FeedForwardBlock(512, 2048)