import os
import tensorflow as tf
import pandas as pd 
# import multiprocessing
from time import time
from loguru import logger
from sklearn.preprocessing import OneHotEncoder
from abstractdecoder.datasets.cfg import CFG
# from multiprocessing import Process, Manager

def get_dataset_path(datatype: str, with_sign=True):
    """
    Args:
    dataset (string): the type of data, only accept either '20k' or '200k'
    with_sign (boolean): if true, using dataset that numbers have been replaced with sign  
    """

    data_dir = "./abstract_to_skim/datasets/pubmed-rct"

    if datatype not in ["20k", "200k"]:
        logger.error("Unsupported dataset...")
        return 

    if datatype == "20k" and with_sign:
        try:
            logger.info("Loading PubMed_20k_RCT_numbers_replaced_with_at_sign dataset...")
            dir = os.path.join(data_dir, "PubMed_20k_RCT_numbers_replaced_with_at_sign")
            return [
                os.path.join(dir, "train.txt"),
                os.path.join(dir, "dev.txt"),
                os.path.join(dir, "test.txt")
            ]
        except FileNotFoundError as e:
            logger.error(e)
    else:
        try:
            logger.info("Loading PubMed_20k_RCT dataset...")
            dir = os.path.join(data_dir, "PubMed_20k_RCT")
            return [
                os.path.join(dir, "train.txt"),
                os.path.join(dir, "dev.txt"),
                os.path.join(dir, "test.txt")
            ]
        except FileNotFoundError as e:
            logger.error(e)

    if datatype == "200k" and with_sign:
        try:
            logger.info("Loading PubMed_200k_RCT_numbers_replaced_with_at_sign dataset...")
            dir = os.path.join(data_dir, "PubMed_200k_RCT_numbers_replaced_with_at_sign")
            return [
                os.path.join(dir, "train.txt"),
                os.path.join(dir, "dev.txt"),
                os.path.join(dir, "test.txt")
            ]
        except FileNotFoundError as e:
            logger.error(e)
    else:
        try:
            logger.info("Loading PubMed_200k_RCT dataset...")
            dir = os.path.join(data_dir, "PubMed_200k_RCT")
            return [
                os.path.join(dir, "train.txt"),
                os.path.join(dir, "dev.txt"),
                os.path.join(dir, "test.txt")
            ]
        except FileNotFoundError as e:
            logger.error(e)

def preprocess_func(returned_values, dir):
    with open(dir, "r") as f:
        lines = f.readlines()
        f.close()

    abstract_lines = ""
    abstract_samples = []

    for line in lines:
        if line.startswith("###"):
            abstract_id = line
            abstract_lines = ""
        elif line.isspace(): # check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines() # split abstract into separate lines

            # Iterate through each line in abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {} # create empty dict to store data from line
                target_text_split = abstract_line.split("\t") # split target label from text
                line_data["target"] = target_text_split[0] # get target label
                line_data["text"] = target_text_split[1].lower() # get target text and lower it
                line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract?
                line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are in the abstract? (start from 0)
                abstract_samples.append(line_data) # add line data to abstract samples list

        else: # if the above conditions aren't fulfilled, the line contains a labelled sentence
            abstract_lines += line

    returned_values.append(abstract_samples)

def preprocessing_data(datatype: str, with_sign=True):
    """
    preprocessing data into this format:
        [
            {"target": 'CONCLUSION',
            "text": The study couldn't have gone better, turns out people are kinder than you think",
            "line_number": 8,
            "total_lines": 8}
        ]

    Args:
        dataset (string): the type of data, only accept either '20k' or '200k'
    """
    start = time()

    data_dirs = get_dataset_path(datatype, with_sign)

    if isinstance(data_dirs, list) == False:
        logger.error("Face error while getting dataset paths.")
        return

    returned_values = []
    for d in data_dirs:
        logger.info(f"Start processing {d.split('/')[-1]}")
        preprocess_func(returned_values, d)

    train_ds, val_ds, test_ds = [x for x in returned_values]

    logger.info(f"Total time: {(time() - start):.2f}s")
    return train_ds, val_ds, test_ds

def get_data_ready(train_ds, 
                   val_ds, 
                   test_ds, 
                   batch_size=CFG.BATCH_SIZE,
                   line_numbers_depth=CFG.LINE_NUMBER_DEPTH,
                   total_line_depth=CFG.TOTAL_LINE_DEPTH
                   ):
    try:
        train_df = pd.DataFrame(train_ds)
        val_df = pd.DataFrame(val_ds)
        test_df = pd.DataFrame(test_ds)

        one_hot_encoder = OneHotEncoder(sparse_output=False)

        # Input 1: Sentences
        train_sentences = train_df['text'].to_numpy()
        val_sentences = val_df['text'].to_numpy()
        test_sentences = test_df['text'].to_numpy()

        # Input 2: Line numbers
        train_line_numbers = tf.one_hot(train_df['line_number'].to_numpy(), depth=line_numbers_depth)
        val_line_numbers = tf.one_hot(val_df['line_number'].to_numpy(), depth=line_numbers_depth)
        test_line_numbers = tf.one_hot(test_df['line_number'].to_numpy(), depth=line_numbers_depth)

        # Input 3: Total line 
        train_total_lines = tf.one_hot(train_df['total_lines'].to_numpy(), depth=total_line_depth)
        val_total_lines = tf.one_hot(val_df['total_lines'].to_numpy(), depth=total_line_depth)
        test_total_lines = tf.one_hot(test_df['total_lines'].to_numpy(), depth=total_line_depth)

        # Target
        train_targets = one_hot_encoder.fit_transform(train_df['target'].to_numpy().reshape(-1, 1))
        val_targets = one_hot_encoder.transform(val_df['target'].to_numpy().reshape(-1, 1))
        test_targets = one_hot_encoder.transform(test_df['target'].to_numpy().reshape(-1, 1))

        # Data pipeline
        train_tribrid_data = tf.data.Dataset.from_tensor_slices(
            (train_sentences, train_line_numbers, train_total_lines)
        )
        train_tribrid_label = tf.data.Dataset.from_tensor_slices(
            train_targets
        )
        train_dataset = tf.data.Dataset.zip(train_tribrid_data, train_tribrid_label).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_tribrid_data = tf.data.Dataset.from_tensor_slices(
            (val_sentences, val_line_numbers, val_total_lines)
        )
        val_tribrid_label = tf.data.Dataset.from_tensor_slices(
            val_targets
        )
        val_dataset = tf.data.Dataset.zip(val_tribrid_data, val_tribrid_label).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        test_tribrid_data = tf.data.Dataset.from_tensor_slices(
            (test_sentences, test_line_numbers, test_total_lines)
        )
        test_tribrid_label = tf.data.Dataset.from_tensor_slices(
            test_targets
        )
        test_dataset = tf.data.Dataset.zip(test_tribrid_data, test_tribrid_label).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        logger.info("Your data is ready for training..")
        return train_dataset, val_dataset, test_dataset
    except Exception as e: 
        logger.error(e)

if __name__ == "__main__":
    train_ds, val_ds, test_ds = preprocessing_data("20k", True)
    train_dataset, val_dataset, test_dataset = get_data_ready(train_ds, val_ds, test_ds)