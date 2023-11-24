import argparse
import json
import numpy as np
import torch
import random
from datasets import Dataset

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# Training, we don't concatenate train and inference arguments for the readability
def parse_train_args():
    parser = argparse.ArgumentParser(
        description='Train a bert classifier on a dataset'
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_save_path", type=str, default="./model_weight/kcbert-base-kold-mlm")
    parser.add_argument("--tensorboard_path", type=str, default="./runs/kcbert-base-kold-mlm")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="beomi/kcbert-base")
    parser.add_argument("--logger_path", type=str, default="train_log.txt")
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

# Inference
def parse_inference_args():
    parser = argparse.ArgumentParser(description="Inference normalized probability with bert")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./kcbert_mlm_trained")
    parser.add_argument("--tokenizer", type=str, default="beomi/kcbert-base")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mask_1_sentence", type=str, default="저 [MASK]는 간호사이다.")
    parser.add_argument("--mask_2_sentence", type=str, default="저 [MASK]는 [MASK]이다.")
    parser.add_argument("--is_first_target", type=bool, default=True, help="The first mask token is the mask token of the target word.")
    parser.add_argument("--bias_type", type=str, default="gender", help="nation or gender")
    args = parser.parse_args()
    return args
