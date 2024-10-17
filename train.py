import pandas as pd
import tensorflow as tf 
from argparse import ArgumentParser
from abstractdecoder.datasets.transform import *
from abstractdecoder.datasets.utils import *
from abstractdecoder.model import TransformerEncoderModel
from abstractdecoder.metrics import ClassificationMetrics
from abstractdecoder.callbacks import *

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-dn", "--dataset_name", type=str, default="20k", help="Name of the dataset that used for training. Only accept 20k or 200k."
    )
    parser.add_argument(
        "-ws", "--with_sign", type=bool, default=True, help="Type of the dataset."
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=32, help="Size of each batch."
    )
    parser.add_argument(
        "-ld", "--line_numbers_depth", type=int, default=15, help="95% percentile line numbers in one training data."
    )
    parser.add_argument(
        "-td", "--total_line_depth", type=int, default=20, help="95% percentile total lines among all training data."
    )
    parser.add_argument(
        "-ed", "--embedding_dim", type=int, default=512, help="Embedding dimension of each token."
    )
    parser.add_argument(
        "-fd", "--ff_dim", type=int, default=2048, help="The number of hidden units in a feed forward layer."
    )
    parser.add_argument(
        "-nh", "--num_heads", type=int, default=4, help="The number of heads in Multi-heads Attention layer."
    )
    parser.add_argument(
        "-dr", "--dropout_rate", type=float, default=0.1, help="The dropout rate."
    )
    parser.add_argument(
        "-vs", "--vocab_size", type=int, default=68000, help="The number of tokens after tokenizing."
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="Training epoch."
    )
    parser.add_argument(
        "-ep", "--early_stopping_patience", type=int, default=5, help="The time needed to stop training if the monitor value stay unchanged."
    )
    parser.add_argument(
        "-lp", "--lr_scheduler_patience", type=int, default=3, help="The time need to reduce the learning rate if the monitor value stay unchanged. This should always smaller than early stopping's patience."
    )
    parser.add_argument(
        "-lf", "--lr_scheduler_factor", type=float, default=0.2, help="The value that used to scale down the learning rate, usually between 0.1 and 0.5."
    )
    parser.add_argument(
        "-m", "--monitor", type=str, default="val_loss", help="The value that is trackd at each epoch."
    )
    parser.add_argument(
        "-o", "--save_dir", type=str, default="./checkpoints/checkpoint.weights.h5", help="Model checkpoint save directory."
    )

    args = parser.parse_args()

    train_ds, val_ds, test_ds = preprocessing_data(args.dataset_name, args.with_sign)
    train_df = pd.DataFrame(train_ds)

    vectorizer = text_vectorization(train_df['text'].to_numpy())
    
    train_dataset, val_dataset, test_dataset = get_data_ready(train_ds, 
                                                              val_ds, 
                                                              test_ds,
                                                              batch_size=args.batch_size,
                                                              line_numbers_depth=args.line_numbers_depth,
                                                              total_line_depth=args.total_line_depth)

    transformer_encoder_model = TransformerEncoderModel(
        vectorizer=vectorizer,
        embed_dim=args.embedding_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        vocab_size=args.vocab_size,
    )

    transformer_encoder_model.compile(
        loss = "categorical_crossentropy",
        optimizer = tf.keras.optimizers.Adam(),
        metrics = [ClassificationMetrics()]
    )

    history = transformer_encoder_model.fit(
        train_dataset,
        epochs = args.epochs,
        validation_data=val_dataset,
        callbacks=[
            early_stopping(patience=args.early_stopping_patience, monitor=args.monitor),
            model_checkpoint(filepath=args.save_dir, monitor=args.monitor),
            learning_scheduler(monitor=args.monitor,
                               patience=args.lr_scheduler_patience,
                               factor=args.lr_scheduler_factor)
        ]
    )