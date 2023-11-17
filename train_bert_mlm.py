import os

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES']='0'
import utils
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW, DataCollatorForLanguageModeling
import logging
import sys
from tqdm import tqdm
from random_seed import set_random_seed

def tokenize_function(ex):
    return tokenizer([sequence for sequence in ex['sentence']], truncation=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = utils.parse_args()
    set_random_seed(random_seed=args.seed)
    handlers = [logging.FileHandler(args.logger_path), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(handlers=handlers, level=logging.INFO)
    
    # print current args
    logging.info(args.__dict__)

    writer = SummaryWriter(args.tensorboard_path)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = BertForMaskedLM.from_pretrained(args.pretrained_model_name_or_path).to(device)

    dataset = utils.prepare_dataset('./kold_v1.json')

    # split dataset into train and test
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=args.seed)

    # tokenize dataset and remove the sentence
    dataset = dataset.map(tokenize_function, batched=True, batch_size=args.batch_size)
    dataset = dataset.remove_columns(["sentence"])

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size, collate_fn=data_collator)


    for epoch in range(args.num_epochs):
        logging.info(f"Epoch : {epoch}\n")
        logging.info("=====Training Phase=====")
        train_loss_sum = 0
        valid_loss_sum = 0
        lowest_loss = sys.float_info.max

        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            writer.add_scalar('Train step loss', loss, 1+i+len(train_dataloader)*epoch)
            print(f"Step: {i+len(train_dataloader)*epoch} Train step loss: "+"{:.2f}".format(loss)+" Epoch percentage: "+"{:.2f}%".format(round(i/len(train_dataloader), 4)*100))

        model.eval()
        logging.info("=====Validation Phase=====")
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            del batch['labels']
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
            valid_loss_sum += loss

        if valid_loss_sum < lowest_loss:
            lowest_loss = valid_loss_sum
            model.save_pretrained(f'./model_weight/bert_trained_{round(lowest_loss, 2)}')

        writer.add_scalar('Train epoch loss', train_loss_sum/len(train_dataloader), epoch)
        writer.add_scalar('Valid epoch loss', valid_loss_sum/len(test_dataloader), epoch)
        logging.info(f"### Train epoch loss: {train_loss_sum/len(train_dataloader)}  ### Valid epoch loss: {valid_loss_sum/len(test_dataloader)}")