# hate Speech Detection

This package detects hate speech and offeisive Tweets using BERT model. Given a tweet or sentence, the model will classify it as hate_tweet, offensive_tweet or normal_tweets.

## Requirements

* torch==1.7.1+cu101
* torchvision==0.8.2+cu101
* transformers==4.6.1
* numpy>=1.18.5
* demoji>=1.1.0

## Installation

To install the package from nexus, use:

~~bash

pip install --extra-index-url=https://nexus.smeir.io/repository/pypi-hosted/simple hateSpeechDetect==1.0 -f https://download.pytorch.org/whl/torch_stable.html

~~

## Usage 

~~python
from hateSpeechDetect import HateSpeechDetector

hd = HateSpeechDetector().from_pretrained('pretrained')
tweets = [ 'Some of these mfs have BLM in their bios. How tf are u gonna be racist and then say BLM? hypocritical',
    'White people sacrificed for US,where is BLM? Jeo should kneel down to them,not BLM',
    'you ready for crazy ass racing tomorrow',
    'what do you expect from the likes of eric erickson he is a piece of crap',
    'meneither she is a liar clear as day a liar url 1',
    'most americans find any of this hard to believe plus yrs later another justice thomas hit job by the media and liberals',
    'gfy the nfl should get no protection from the police and no support from our military the league will slowly die on this road of shame',
    'invite to all of the shows and let her tell you about her interview with brock long then you will know why cult politburo sounds so unhinged
 he is parroting the fema admins line of bs tried to fight back against long s bs',
    'forget michael he is a coward',
    'that s not am buller point on the list of voters needs wiunion wipolitics wiright maga nra democrats demonicrats pervertsinc url',
    'he gets confirmed conservatives don t care that he s a rapist at all',
    'Some of these mfs have BLM in their bios. How tf are u gonna be racist and then say BLM? hypocritical']

hate_tweets, offensive_tweets, normal_tweets = hd.predict(tweets)
print("hate speech : ", hate_tweets)
print("offensive speech : ", offensive_tweets)
print("normal speech : ", normal_tweets)
~~

The expected output:

~~

hate speech :  ['Some of these mfs have BLM in their bios. How tf are u gonna be racist and then say BLM? hypocritical', 'White people sacrificed for US,where is BLM? Jeo should kneel down to them,not BLM', 'Some of these mfs have BLM in their bios. How tf are u gonna be racist and then say BLM? hypocritical']
offensive speech :  ['you ready for crazy ass racing tomorrow', 'what do you expect from the likes of eric erickson he is a piece of crap', 'meneither she is a liar clear as day a liar url 1', 'gfy the nfl should get no protection from the police and no support from our military the league will slowly die on this road of shame', 'forget michael he is a coward', 'he gets confirmed conservatives don t care that he s a rapist at all']
normal speech:  ['most americans find any of this hard to believe plus yrs later another justice thomas hit job by the media and liberals', 'invite to all of the shows and let her tell you about her interview with brock long then you will know why cult politburo sounds so unhinged he is parroting the fema admins line of bs tried to fight back against long s bs', 'that s not am buller point on the list of voters needs wiunion wipolitics wiright maga nra democrats demonicrats pervertsinc url']
~~
## Build and Deploy

To build and deploy python library to Nexus, use the deploy script:

~~bash
bash run_deploy.sh
~~

This automatically copies over checkpoint files as part of the python package. 

## Evaluation with HateSpeechData.csv

Accuracy: 92.0%
Macro-averaged f1 score: 73.3%
Macro-averaged average prevision: 80.6%

## Author

~~
Author: DW Shin
Email: dwshin@idsinternational.com
~~
