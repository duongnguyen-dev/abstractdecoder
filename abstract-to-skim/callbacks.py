import tensorflow as tf

def early_stopping(patience=5, monitor="val_loss"):
    """
    patience: the time needed to stop training if the monitor value stay unchanged
    monitor: the value used to keep track at each epoch
    """
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)

def model_checkpoint(filepath, monitor="val_loss"):
    """
    filepath: Dir path to save the model's checkpoint
    monitor: the value used to keep track at each epoch
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

def learning_scheduler(monitor="val_loss", factor=0.2, patience=5):
    """
    monitor:  monitor: the value used to keep track at each epoch
    factor: the value that used to scale down the learning rate, usually between 0.1 and 0.5
    patience: the time need to reduce the learning rate if monitor value stay unchanged
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        verbose=1   
    )
