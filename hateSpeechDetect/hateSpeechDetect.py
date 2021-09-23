import os, sys, re, itertools, glob, time, random
import emoji
from urllib.parse import urlparse
from datetime import datetime
from scipy.special import softmax
import numpy as np
import tensorflow as tf
from transformers import AutoModel, BertTokenizerFast, TrainingArguments, Trainer, BertForSequenceClassification
from .utils import *
import torch
import demoji
demoji.download_codes()

ws_cleaner = re.compile(r'\s+', re.MULTILINE)
url_cleaner = re.compile(r'(?:http|www\.)\S+')
mention_cleaner = re.compile(r'\@[A-Za-z0-9_]+')
hashtag_cleaner = re.compile(r'\#[A-Za-z0-9_]+')

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
        self.base_weights = 'bert-base-uncased'
        self.initial_epoch = 0

        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.model = self.bert_model

        self.categories = [0,1,2]

        self._train_generator = None
        self._tune_generator = None
        self._dev_generator = None
        self._test_generator = None

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

    def from_pretrained(self, ckpt_dir):
        full_path = os.path.join(self.root_dir, ckpt_dir)
        
        print(full_path)
        if os.path.exists(full_path):
            fpath = full_path
        else:
            raise Exception('No such file exists: {}'.format(ckpt_dir))

        print('Checking for checkpoint at: {}'.format(fpath))
        bert_model = BertForSequenceClassification.from_pretrained(fpath)
        tokenizer = BertTokenizerFast.from_pretrained(self.base_weights)
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
