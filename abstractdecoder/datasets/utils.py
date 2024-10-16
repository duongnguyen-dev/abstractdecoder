import numpy as np
import tensorflow as tf 
from abstractdecoder.datasets.cfg import CFG

def find_output_sequences_length(sequences):
    sequences_length = [len(s.split()) for s in sequences]
    return int(np.percentile(sequences_length, 95))

def text_vectorization(sequences, max_tokens=CFG.MAX_TOKENS):
    output_sequence_length = find_output_sequences_length(sequences)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens = max_tokens,
        output_sequence_length = output_sequence_length,
        pad_to_max_tokens = True,
        name = "TextVectorizationLayer"
    )

    vectorizer.adapt(sequences)
    return vectorizer