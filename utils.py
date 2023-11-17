import argparse
import json
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a bert classifier on a dataset'
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_path", type=str, default="./model_weight")
    parser.add_argument("--tensorboard_path", type=str, default="./runs/bert_for_classification")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--logger_path", type=str, default="log.txt")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)


    args = parser.parse_args()
    return args

def prepare_dataset(path_name):
    # load kold data fetched from :
    # https://github.com/boychaboy/KOLD/blob/main/data/kold_v1.json
    with open(path_name, mode='r', encoding='utf-8') as file:
        json_file = file.read()
    json_list = json.loads(json_file)

    # concat title and comment and append to dataset list
    # we simply do a contatenation of title and comment as a sentence
    dataset_list = [data['title'] + ' ' + data['comment'] for data in json_list]
    
    # dataset_list to Dataset
    def gen():
        for i in dataset_list:
            yield {"sentence": i}

    return Dataset.from_generator(generator=gen)
