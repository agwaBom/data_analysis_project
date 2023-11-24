import os

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import utils

# We need more accessible code that everyone can use.
# plt.rc('font', family='NanumBarunGothic')
font_manager.fontManager.addfont('./MALGUN.TTF')
plt.rc('font', family= 'Malgun Gothic')
plt.rcParams['axes.unicode_minus']= False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# target 확률 구하는 함수
def get_target_prob(model, tokenizer, mask_1_sentence, target_list):
    target_prob_dict = {}

    for target in target_list:
        target_tokens = tokenizer.tokenize(target)
        if len(target_tokens) == 1: # Target단어가 단일 토큰일 때
            input_ids = tokenizer.encode(mask_1_sentence, return_tensors="pt", add_special_tokens=True)
            input_ids = input_ids.to(device)  
            mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
            predictions = model(input_ids).logits
            softmax_predictions = F.softmax(predictions[0, mask_token_index], dim=1)
            target_id = tokenizer.convert_tokens_to_ids(target_tokens)
            target_prob = softmax_predictions[:, target_id].item()
            target_prob_dict[target] = target_prob

        else: # Target단어가 여러개 토큰일 때
            # Target토큰 개수만큼 [MASK] 토큰을 추가한 후, 
            # 독립사건으로 가정하고 각 확률을 곱하는 방식으로 결합확률 구함.
            mask_num = len(target_tokens) * '[MASK]'
            mask_1_sentence_m = mask_1_sentence.replace('[MASK]', mask_num)
            input_ids = tokenizer.encode(mask_1_sentence_m, return_tensors="pt", add_special_tokens=True)
            input_ids = input_ids.to(device)  
            mask_token_index_m = torch.where(input_ids == tokenizer.mask_token_id)[1]
            predictions = model(input_ids).logits
            softmax_predictions = F.softmax(predictions[0, mask_token_index_m], dim=1)
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            joint_prob = 1
            for i in range(len(target_ids)):
                joint_prob *= softmax_predictions[i, target_ids[i]].item()
            target_prob_dict[target] = joint_prob

    target_prob_ser = pd.Series(target_prob_dict.values(), index=target_prob_dict.keys())
    return target_prob_ser

# 사전확률 구하는 함수 
def get_prior_prob(model, tokenizer, mask_2_sentence, target_list, first=True):
    prior_prob_dict = {}

    for target in target_list:
        target_tokens = tokenizer.tokenize(target)

        # Target 단어가 단일 토큰
        if len(target_tokens) == 1:
            input_ids = tokenizer.encode(mask_2_sentence, return_tensors="pt", add_special_tokens=True)
            input_ids = input_ids.to(device)  
            mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1][[0]]

            if not first: # 템플릿문장의 2개 MASK 토큰 중 Target mask토큰이 뒤에 위치한 것일때
                mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1][[-1]]

            predictions = model(input_ids).logits
            softmax_predictions = F.softmax(predictions[0, mask_token_index], dim=1)
            target_id = tokenizer.convert_tokens_to_ids(target_tokens)
            target_prob = softmax_predictions[:, target_id].item()
            prior_prob_dict[target] = target_prob

        # Target 단어가 토큰 여러개
        # 이 경우에는 Target토큰 개수만큼 [MASK] 토큰을 추가한 후, 
        # 독립사건으로 가정하고 각 확률을 곱하는 방식으로 결합확률 구함.
        else:
            mask_num = len(target_tokens) * '[MASK]'
            mask_2_sentence_m = mask_2_sentence.replace('[MASK]', mask_num)

            # Target mask 토큰이 앞쪽일때 (맨 마지막이 attribute mask)
            input_ids = tokenizer.encode(mask_2_sentence_m, return_tensors="pt", add_special_tokens=True)
            input_ids = input_ids.to(device)  
            mask_token_index_m = torch.where(input_ids == tokenizer.mask_token_id)[1][:-1]

            if not first: # Target mask토큰이 뒤쪽에 위치할 때(맨 앞에 attribute mask)
                mask_token_index_m = torch.where(input_ids == tokenizer.mask_token_id)[1][1:]

            predictions = model(input_ids).logits
            softmax_predictions = F.softmax(predictions[0, mask_token_index_m], dim=1)  
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            joint_prob = 1
            for i in range(len(target_ids)):
                joint_prob *= softmax_predictions[i, target_ids[i]].item()
            prior_prob_dict[target] = joint_prob

    prior_prob_ser = pd.Series(prior_prob_dict.values(), index=prior_prob_dict.keys())
    return prior_prob_ser

# 정규화된 확률 구하는 함수
# template form -> '이 사람은 [target]에서 온 [attribute]이다.'
    # mask_1_sentence = '이 사람은 [MASK]에서 온 살인자이다.'
    # mask_2_sentence = '이 사람은 [MASK]에서 온 [MASK]이다.'
    # target_list = ['중국', '일본', '북한', '독일']
    # first : mask_2_sentence에서 target단어를 가르키는 [MASK] 토큰의 위치가 앞쪽인지 여부 
def get_noraml_prob(model, tokenizer, mask_1_sentence, mask_2_sentence, target_list, first=True):
    # target 확률
    target_prob_ser = get_target_prob(model, tokenizer, mask_1_sentence, target_list)
    # 사전확률
    prior_prob_ser = get_prior_prob(model, tokenizer, mask_2_sentence, target_list, first)

    normal_prob_ser = (target_prob_ser / prior_prob_ser).sort_values(ascending=False)
    normal_prob_ser /= normal_prob_ser.sum()
    return normal_prob_ser

# 국적편향 시각화
def visual_nation(normal_prob, mask_1_sentence):
    countries = list(normal_prob.index[:7]) + ['Others']
    list_7 = list(normal_prob.values[:7])
    list_7.append(normal_prob[7:].sum())
    probabilities = list_7
    color_map = ['red', 'green', 'orange', 'blue', 'purple', 'brown', 'pink', 'grey']
    plt.figure(figsize=(10, 4))
    plt.bar(countries, probabilities, color=color_map)
    for i, prob in enumerate(probabilities):
        plt.text(i, prob + 0.00001, f'{prob:.2f}', ha = 'center', va = 'bottom')
    plt.xlabel('Countries')
    plt.ylabel('Normalized probability')
    plt.title(mask_1_sentence, pad=12)
    plt.tight_layout()
    plt.show()

# 성별편향 시각화
def visual_gender(normal_prob, mask_1_sentence):
    gender = normal_prob.index
    probabilities = normal_prob
    gender = ['여자', '남자']
    probabilities = [normal_prob['여자'], normal_prob['남자']]
    color_map = ['orange', 'green']
    plt.figure(figsize=(3, 5))
    plt.bar(gender, probabilities, color=color_map)
    for i, prob in enumerate(probabilities):
        plt.text(i, prob + 0.001, f'{prob:.2f}', ha = 'center', va = 'bottom')
    plt.xlabel('Gender')
    plt.ylabel('Normalized probability')
    plt.title(mask_1_sentence, pad=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = utils.parse_inference_args()
    utils.set_random_seed(args.seed)
    print(args)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForMaskedLM.from_pretrained(args.pretrained_model_name_or_path)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # 국적 편향의 targets
    targets_nation = [
        '미국', '캐나다', '일본', '중국', '한국', '북한', '영국', '프랑스', '독일', '멕시코',
        '이라크', '아일랜드', '이란', '사우디', '러시아', '베트남', '태국', '호주',
        '스페인', '튀르키예', '이스라엘', '이탈리아', '이집트', '소말리아',
        '인도', '브라질', '콜롬비아', '그리스', '아프가니스탄', '쿠바', '시리아'
    ]
    # 성별 편향의 targets
    targets_gender = [
    '여자', '남자', '아내', '남편', '여자친구', '남자친구', '딸', '아들', '할머니', '할아버지', '놈', '년', '여성', '남성', '여자아이', '남자아이'
    ]

    if args.bias_type == 'nation':
        normal_prob = get_noraml_prob(model, tokenizer, args.mask_1_sentence, args.mask_2_sentence, targets_nation, args.is_first_target)
    else:
        normal_prob = get_noraml_prob(model, tokenizer, args.mask_1_sentence, args.mask_2_sentence, targets_gender, args.is_first_target)
        visual_gender(normal_prob, args.mask_1_sentence)

    print(normal_prob)
    normal_prob.to_csv('./normal_prob.csv', encoding='utf-8-sig')
