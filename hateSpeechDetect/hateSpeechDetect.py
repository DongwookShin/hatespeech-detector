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

    '''
    def build_model(self, training_mode=False, fixed_encoder=False):
        #emojis = list(emoji.unicode_codes.EMOJI_UNICODE_ENGLISH.keys())
        #emojis += list(emoji.unicode_codes.EMOJI_ALIAS_UNICODE_ENGLISH.keys())
        #special_tokens = ['#HASH', '@USER', '<NEXT>', '&amp;'] + sorted(set(emojis))

        self.tokenizer = BertTokenizerFast.from_pretrained(self.base_weights)
        #self.tokenizer.add_tokens(special_tokens)

        self.bert_model = BertForSequenceClassification.from_pretrained(self.base_weights)
        #self.bert_model.resize_token_embeddings(len(self.tokenizer))

        # self.model = self._construct_base_model(self.bert_model, len(self.categories), training_mode=training_mode, fixed_encoder=fixed_encoder)
        self.model = self.bert_model
        self.model.summary()
    
    def _construct_base_model(self, bert_model, num_outputs, training_mode, fixed_encoder):
        input_ids = tf.keras.Input([None], dtype='int32', name='input_ids')
        attention_mask = tf.keras.Input([None], dtype='int32', name='attention_mask')
        token_type_ids = tf.keras.Input([None], dtype='int32', name='token_type_ids')

        input_dict = { 'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask }

        bmodel = bert_model(input_dict, training=training_mode)

        pooled_output = bmodel.last_hidden_state[:, 0, :]

        dropout_layer = tf.keras.layers.Dropout(0.5)
        dense_layer_inner = tf.keras.layers.Dense(512, activation='tanh')
        dense_layer_outter = tf.keras.layers.Dense(num_outputs)
        output = dense_layer_outter(dropout_layer(dense_layer_inner(pooled_output)))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2)
        cata = tf.keras.metrics.CategoricalAccuracy(name='exact_accuracy')
        model = tf.keras.Model(inputs=input_dict, outputs=output)
        
        if fixed_encoder:
            for k, layer in enumerate(model.layers):
                if layer.name == 'tf_bert_model': 
                    layer.trainable = False
                    print('Disabled training for layer: {}'.format(layer.name))

        model.compile(optimizer=optimizer, loss=cce_loss, metrics=[cata])

        return model
    '''

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

    '''
    def train(self, ckpt_name, num_epochs=None, batch_size=128, steps_per_epoch=1000, quiet=False, log=False):
        if self._train_generator is None:
            raise Exception('Must set data pointer before training!')

        train_dataset = self._create_generator(self._train_generator, batch_size)

        if self._dev_generator is None:
            print('WARNING: Dev set generator not set. No validation results will be printed.')
            dev_dataset = None
        else:
            dev_dataset = self._create_generator(self._dev_generator, batch_size).repeat().take(50)

        if quiet:
            verbose = 2
        else:
            verbose = 1

        ckpt_callback = ModelPeriodicCheckpoint(freq=4, ckpt_path=(self.get_checkpoint_basepath(ckpt_name)))
        callback_list = [ckpt_callback]
        if log:
            logdir = os.path.join('logs/', datetime.now().strftime('%Y%m%d-%H%M%S'))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            callback_list.append(tensorboard_callback)

        print('** Training model...')
        self.model.fit(train_dataset, validation_data=dev_dataset, epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
            initial_epoch=(self.initial_epoch - 1),
            callbacks=callback_list)
    '''
    
    def predict(self, org_tweets, return_probs=False):
        processed_text = HateSpeechDetector._preprocess_text(org_tweets)
        # print("self.bert_model : ", self.bert_model)
        trainer = Trainer( model  = self.bert_model)
        # trainer = Trainer( model  = model)

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
           elif y_pred == 2: # offensive
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
