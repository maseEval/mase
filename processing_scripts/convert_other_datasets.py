#!/usr/bin/env python
# coding: utf-8
#
# This script converts datasets like Snips and Pico into the FSC data format.
#
# Usage:
# python processing_scripts/convert_other_datasets.py --data snips --recording-type close-field \
# --output-directory snips_slu_data_v1.0/close_field_splits
# or
# python processing_scripts/convert_other_datasets.py --data pico --train-ratio 0.7 --test-ratio 0.2 \
# --output-directory data_pico/splits

import pandas as pd
import argparse
import json
import pdb
import os
from sklearn.model_selection import train_test_split
import string

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        print(f"Creating directory at {path}")
        os.mkdir(path)


def write_file(df, path):
    df.to_csv(path, index = False)
    print(f"Wrote file to {path}.")


def create_splits(df, output_directory, train_ratio=0.8, test_ratio=0.1, random_state=42):
    '''
    Adapted from https://github.com/alexa/alexa-end-to-end-slu/blob/main/dataprep/prepare_snips.py.
    Creates train, test, and validation splits using the provided ratios.
    '''
    df_valtest, df_train = train_test_split(df, test_size=train_ratio, random_state=random_state)
    test_to_val_ratio = test_ratio / (1 - train_ratio)
    df_val, df_test = train_test_split(df_valtest, test_size=test_to_val_ratio, random_state=random_state)
    create_dir_if_not_exists(output_directory)

    write_file(df_train, os.path.join(output_directory, "train_data.csv"))
    write_file(df_test, os.path.join(output_directory, "test_data.csv"))
    write_file(df_val, os.path.join(output_directory, "valid_data.csv"))

def get_pico_data(base_dir, speech_folder='data_pico/speech/clean', out_directory='data_pico/splits'):
    data_file = 'data_pico/label/label.json'
    with open(data_file, 'rb') as f:
        all_labels = json.load(f)
    keys = [k for k in all_labels]
    rel_path = os.path.relpath(speech_folder, base_dir)
    full_paths = [os.path.join(rel_path,k) for k in keys]
    labels = [all_labels[k]['slots']['coffeeDrink'] for k in keys]
    labels_set = sorted(set(labels))
    labels_unique = {lbl: i for i, lbl in zip(range(len(labels_set)), labels_set)}
    
    print("Pico labels: ")
    print(labels_unique)
    
    labels_num = [labels_unique[lbl] for lbl in labels]

    assert len(full_paths) == len(labels)
    assert len(labels) == len(labels_num)

    deleted_rows = 0
    for i in reversed(range(len(full_paths))):
        abspath = os.path.join(base_dir, full_paths[i])
        if not os.path.exists(abspath):
            del full_paths[i]
            del labels[i]
            del labels_num[i]
            deleted_rows += 1

    print(f"Deleted {deleted_rows} from Pico with missing audio files. {len(full_paths) - deleted_rows} remain.")

    data_list = pd.DataFrame.from_dict({'path': full_paths, 'intent': labels, 'intentLbl': labels_num})
    data_list.to_csv(os.path.join(base_dir, 'pico_labeled.csv'), index = False)
    return data_list

SNIPS_RECORDING_TYPES = ["close-field", "far-field"]
def get_snips_data(base_dir, speech_folder = 'snips_slu_data_v1.0', recording_type="close-field"):
    if recording_type not in SNIPS_RECORDING_TYPES:
        raise ValueError(f"Recording type {recording_type} not supported; must be one of {SNIPS_RECORDING_TYPES}")
    folder_ = f"smart-lights-en-{recording_type}"
    print(f"Consuming data from {folder_}")
    datapath = os.path.join(speech_folder, folder_)
    with open(os.path.join(datapath, 'dataset.json'), 'rb') as f:
        intents_list = json.load(f)

    all_intents= list(intents_list['intents'].keys())
    intents_set = sorted(set(all_intents))
    labels_unique = {lbl: i for i, lbl in zip(range(len(intents_set)), intents_set)}
    
    text_to_intent = {}
    for intent in all_intents:
        utterances = intents_list['intents'][intent]['utterances']
        for i in range(len(utterances)):
            texts = [j['text'] for j in utterances[i]['data']]
            texts = ''.join(texts)
            text_to_intent[texts] = intent
            
    with open(os.path.join(datapath, 'speech_corpus', 'metadata.json'), 'rb') as f:
        meta = json.load(f)

    # The training script loads speech files from a relative path, which is the part of the
    # splits directory before "/data/". Therefore, we represent our data path as a relative
    # path, to be compatible with the training code.
    full_paths = []
    sp_ids = []
    labels = []
    transcripts = []
    rel_path = os.path.relpath(datapath, base_dir)

    for data_ in meta.values():
        fname = data_['filename']
        text_ = data_['text']
        speaker_id = data_['worker']['id']
        intent_ = text_to_intent[text_]
        
        full_paths.append(os.path.join(rel_path, 'speech_corpus', 'audio', fname))
        sp_ids.append(speaker_id)
        labels.append(intent_)
        transcripts.append(text_)
    
    print("Snips labels: ")
    print(labels_unique)
    labels_num = [labels_unique[lbl] for lbl in labels]
    data_list = pd.DataFrame.from_dict({'path': full_paths,
                                        'speakerId': sp_ids,
                                        'transcription': transcripts,
                                        'intent': labels,
                                        'intentLbl': labels_num})
    data_list.to_csv(os.path.join(base_dir, f'snips_labeled_{recording_type}.csv'), index = False)
    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="indicate if pico or snips", required=True)
    parser.add_argument("--output-directory", help="where to put splits directory")
    parser.add_argument("--recording-type", default="close-field", help="for Snips, whether to use the close-field or far-field recorded dataset")
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--test-ratio", default=0.1, type=float)
    parser.add_argument("--splits-dir-name", default="original_splits")

    # Read arguments from the command line
    args = parser.parse_args()
    dataset = args.data

    splits_directory = os.path.join(args.output_directory, "data", args.splits_dir_name)

    if args.data == 'pico':
        dataset = get_pico_data(base_dir = args.output_directory)
    else:
        dataset = get_snips_data(recording_type = args.recording_type, base_dir = args.output_directory)
    create_splits(dataset, splits_directory, train_ratio=args.train_ratio, test_ratio=args.test_ratio)

