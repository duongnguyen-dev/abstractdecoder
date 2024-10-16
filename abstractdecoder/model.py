import tensorflow as tf
import pandas as pd
from abstractdecoder.modeling.attention_block import AttentionBlock
from abstractdecoder.modeling.embedding_layer import PositionEmbeddingLayer
from abstractdecoder.cfg import ModelCFG
from abstractdecoder.datasets.utils import text_vectorization
from abstractdecoder.datasets.transform import preprocessing_data, get_data_ready

class TransformerEncoderModel(tf.keras.Model):
    def __init__(self, 
                 vectorizer=None,
                 embed_dim=ModelCFG.EMBED_DIM, 
                 ff_dim=ModelCFG.FF_DIM, 
                 num_heads=ModelCFG.NUM_HEADS, 
                 dropout_rate=ModelCFG.DROPOUT_RATE, 
                 vocab_size=ModelCFG.MAX_TOKENS, 
                ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = ModelCFG.NUM_CLASSES

        self.positional_embedding = PositionEmbeddingLayer(vocab_size, embed_dim)
        self.attention_block = AttentionBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        self.token_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.token_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.line_number_dense = tf.keras.layers.Dense(32, activation="relu", name="LineNumberLayer")
        self.total_line_dense =  tf.keras.layers.Dense(32, activation="relu", name="TotalLineLayer")
        self.concatenate = tf.keras.layers.Concatenate(name="tribrid_embed")
        self.output_dense = tf.keras.layers.Dense(ModelCFG.NUM_CLASSES, activation="softmax")
        self.vectorizer = vectorizer

    def call(self, inputs):
        input_1, input_2, input_3 = inputs

        # Text
        token_output = self.vectorizer(input_1)
        token_output = self.positional_embedding(token_output)
        token_output = self.attention_block(token_output)
        token_output = self.token_average_pooling(token_output)
        token_output = self.token_dropout(token_output)

        # Line number
        line_number_output = self.line_number_dense(input_2)
        
        # Total line
        total_line_output = self.total_line_dense(input_3)
        combined_all = self.concatenate([token_output, line_number_output, total_line_output])

        return self.output_dense(combined_all)
    
if __name__ == "__main__":
    train_ds, val_ds, test_ds = preprocessing_data("20k", True)
    train_df = pd.DataFrame(train_ds)

    vectorizer = text_vectorization(train_df['text'].to_numpy())
    
    train_dataset, val_dataset, test_dataset = get_data_ready(train_ds, val_ds, test_ds)

    transformer_encoder_model = TransformerEncoderModel(
        vectorizer=vectorizer
    )

    result = transformer_encoder_model((tf.constant(["This is test sentence"]), tf.constant([[12]]), tf.constant([[15]])))
