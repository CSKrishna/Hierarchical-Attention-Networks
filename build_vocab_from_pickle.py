"""Build vocabularies of words from datasets"""

import argparse
from collections import Counter
import json
import os
import sys
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")

# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1 # number of buckets (= number of ids) for unknown words
PAD_WORD = '<pad>'



def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(path_pickle, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to csv file
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """    
    data_train = pd.read_pickle(path_pickle)
    docs = data_train[["Sent"]].values.tolist()
    for doc in docs:
        doc = doc[0]
        for sentence in doc:
            vocab.update(str(sentence).split(" "))    
    return len(docs)    


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train, dev, test datasets
    print("Building word vocabulary...")
    words = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train.pkl'), words)
    size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'dev.pkl'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test.pkl'), words)
    print("- done.")
   
   
    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= 6]
    
    # Add pad tokens
    if PAD_WORD not in words: words.append(PAD_WORD)
   
    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences, 
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words) + NUM_OOV_BUCKETS,       
        'pad_word': PAD_WORD,      
        'num_oov_buckets': NUM_OOV_BUCKETS
    }
    
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))