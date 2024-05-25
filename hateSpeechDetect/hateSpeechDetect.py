import os, sys, re, itertools, glob, time, random
from urllib.parse import urlparse
from datetime import datetime
import numpy as np
# import tensorflow as tf
from transformers import AutoModel, BertTokenizerFast, TrainingArguments, Trainer, BertForSequenceClassification
# from .utils import *
import torch
import pkg_resources
import demoji
demoji.download_codes()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#  Create The Dataset Class.
class TheDataset(torch.utils.data.Dataset):

    def __init__(self, texts, values, tokenizer):
        self.texts    =  texts
        self.values = values
        self.tokenizer  = tokenizer
        self.max_len    = tokenizer.model_max_length

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        text = str(self.texts[index])
        value = self.values[index]

        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens    = True,
            max_length            = self.max_len,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors        = "pt",
            padding               = "max_length",
            truncation            = True
        )

        return {
            'input_ids': encoded_text['input_ids'][0],
            'attention_mask': encoded_text['attention_mask'][0],
            'labels': torch.tensor(value, dtype=torch.long)
        }


class HateSpeechDetector(object):

    def __init__(self, root_dir='.', max_length=128, learning_rate=0.0001, tokenizer=None, bert_model=None, augment_data=False):
        self.root_dir = root_dir
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.initial_epoch = 0

        self.tokenizer = tokenizer
        self.bert_model = bert_model


    def train(self, trainfile_name, valid_ratio, ckpt_name, num_epochs=10, batch_size=128, steps_per_epoch=1000, quiet=False, log=False):
        df = pd.read_csv(trainfile_name, sep='\t')
        df = df.dropna()
        # split train dataset into train, validation and test sets
        train_text, valid_text, train_labels, valid_labels = train_test_split(df['text'], df['label'],
                                                                    random_state=2018,
                                                                    test_size=valid_ratio,
                                                                    stratify=df['label'])

        train_dataset = TheDataset(
            texts    = train_text.tolist(),
            values = train_labels.tolist(),
            tokenizer  = self.tokenizer,
        )

        valid_dataset = TheDataset(
            texts    = valid_text.tolist(),
            values = valid_labels.tolist(),
            tokenizer  = self.tokenizer,
        )

        training_args = TrainingArguments(
            output_dir                  = ckpt_name,
            num_train_epochs            = num_epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            warmup_steps                = 500,
            weight_decay                = 0.01,
            save_strategy               = "epoch",
            evaluation_strategy         = "steps"
        )

        trainer = Trainer(
            model           = self.bert_model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = valid_dataset,
            compute_metrics = _compute_metrics
        )

        trainer.train()
    
    def predict(self, org_tweets, return_probs=False):
        processed_text = HateSpeechDetector._preprocess_text(org_tweets)
        trainer = Trainer( model  = self.bert_model)

        test_dataset = TheDataset(
           texts    = processed_text,
           values = [1]*len(processed_text),
           tokenizer  = self.tokenizer
        )
        # Make prediction
        raw_preds, _, _ = trainer.predict(test_dataset)
        # Preprocess raw predictions
        y_preds = np.argmax(raw_preds, axis=1)

        hate_tweets = []
        offensive_tweets = []
        normal_tweets = []

        for idx, y_pred in enumerate(y_preds):
           if y_pred == 0: #hate sppech
               hate_tweets.append(org_tweets[idx])
           elif y_pred == 1: # offensive
               offensive_tweets.append(org_tweets[idx])
           elif y_pred == 2: # normal
               normal_tweets.append(org_tweets[idx])

        return hate_tweets, offensive_tweets, normal_tweets

    @staticmethod
    def from_pretrained():
        path_lookup = pkg_resources.resource_filename(__name__, 'pretrained') 
        # print('Checking for checkpoint at: {}'.format(path_lookup))
        if os.path.exists(path_lookup):
            fpath = path_lookup
        else:
            raise Exception('No such file exists: {}'.format(fpath))

        print('Checking for checkpoint at: {}'.format(fpath))
        bert_model = BertForSequenceClassification.from_pretrained(fpath)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        return HateSpeechDetector(bert_model=bert_model,tokenizer=tokenizer)

    @staticmethod
    def _preprocess_text(tweets):
        newtweets = []
        for text in tweets:
            #text = remove_emoji(text)
            text = demoji.replace(text, "")
            text = text.lower()
            text = re.sub(r'http\S+', '', text)   # removel all URLs
            text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove @ mentions
            # text = re.sub(r'#[a-zA-Z0-9_]+', '', text)  # Remove hashtags
            text = re.sub('([#])|([^a-zA-Z])',' ',text ) # Remove # from hashtag
            text = text.strip(" ")   # Remove whitespace resulting from above

            text = re.sub(r'~[^~].', '', text)  # Remove token starting with ~

            # Handle common HTML entities
            text = re.sub(r'&lt;', '<', text)
            text = re.sub(r'&gt;', '>', text)

            text = re.sub(r'&amp;', '&', text)
            text = re.sub(r'[''?&!\;\:\""-.]', '', text)
            text = re.sub(r' +', ' ', text)   # Remove redundant spaces
            text = re.sub(r'\s+rt\s+', '', text) # remove rt
            text= re.sub(r'\d+(\.\d+)?','number', text)
            text= re.sub(r'[\n\r]',' ', text)

            newtweets.append(text)
        return newtweets

    @staticmethod
    def _compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
