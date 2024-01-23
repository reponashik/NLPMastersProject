import numpy as np
import pandas as pd
from pycorenlp import StanfordCoreNLP
import json, sys, re
import requests
from bs4 import BeautifulSoup



# Getting Positive and Negative words list
pos_words_path = 'positive_words.txt'
neg_words_path = 'negative_words.txt'

file = open(neg_words_path, 'r')
content = file.readlines()
file.close()
neg_words = [word.replace('\n', '') for word in content]

file = open(pos_words_path, 'r')
content = file.readlines()
file.close()
pos_words = [word.replace('\n', '') for word in content]


# Extract content by web scraping
def extract_content(page_link, tags):
    page = requests.get(page_link)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    data = []
    discard = []
    for tag in tags:
        dom = soup.select(tag)
        
        for text in dom:
            text = str(text)
            try:
                while(text.find('>') != -1):
                    s = text.find('<')
                    e = text.find('>')
                    text = text[:s] + text[e+1:]
                data.append(text)
            except Exception as e:                 
                discard.append(text)
                
    # print("Total", len(data), "sentences obtained.")
    if len(discard) > 0:
        print("Total", len(discard), "sentences are discared due to some error in", page_link)

    return data, discard


# Filtering/Cleaning data
def remove_replyinfo(sent):
    if(sent.find("talk") or sent.find("reply") or sent.find("UTC")):
        sent = sent.replace("(talk)", "")
        sent = sent.replace("[reply]", "")
        sent = sent.replace("(UTC)", "")
        
        sent = re.sub("[^\s]+[^\S]+[\d{2}]+[:]+[\d{2}]+[,]+[^\S]+[\d{1,2}]+[^\S]+[a-zA-z]+[^\S]+[\d{4}]", "", sent)
        sent = re.sub("\d{3}", "", sent)
        
        sent = re.sub("[\d{2}]+[:]+[\d{2}]+[,]+[^\S]+[\d{1,2}]+[^\S]+[a-zA-z]+[^\S]+[\d{4}]", "", sent)
        
    return sent


def remove_css(sent):
    sent = sent.replace("-", "")
    
    s = sent.find("{")
    e = sent.find("}")
    
    if s != -1 and e != -1:
        temp = ' '.join(sent[:s].split()[:-1])
        sent = temp + sent[e+1:]

        sent = re.sub('[a-z]+["\-"]+[a-z]', "", sent)
    
    return sent


def remove_punc(sent):
    sent = re.sub("[^a-zA-Z0-9\'\.\?]", ' ', sent)
    sent = ' '.join(sent.split())
    sent = sent.lower()

    return sent


def intersection(list1, list2):
    return set(list1) & set(list2)


def filter_data(data, length):
    data = [remove_replyinfo(sent) for sent in data]
    data = [remove_css(sent) for sent in data]
    data = [remove_punc(sent) for sent in data]
    data = [sent for sent in data if len(sent.split()) > length]
    data = list(set(data))
    return data


# Perform Sentiment Analysis on data
def annotate_data(data):
    nlp = StanfordCoreNLP('http://localhost:9000')
    res = []

    for sent in data:
        
        annotated = nlp.annotate(sent,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 100000,
                       })

        if type(annotated) is str:
            annotated = json.loads(annotated, strict=False)

        pos_count = 0
        neg_count = 0

        for s in annotated['sentences']:
            sent_list = [word['word'] for word in s['tokens']]
            if len(sent_list) > 1:
                if (s['sentiment'] == 'Negative' or s['sentiment'] == 'Verynegative') and len(intersection(neg_words, sent.split())) > 0:
                    neg_count += 1
                elif (s['sentiment'] == 'Positive' or s['sentiment'] == 'Verypositive') and len(intersection(pos_words, sent.split())) > 0:
                    pos_count += 1

        if (neg_count - pos_count) > 0:
            res.append([sent, 'Negative'])
        elif (pos_count - neg_count) > 0:
            res.append([sent, 'Positive'])
        else:
            res.append([sent, 'Neutral'])
        
    return pd.DataFrame(res, columns = ['sentence', 'sentiment'])


def calc_percentage(df, label):
    return round(len(df[df['sentiment'] == label])/len(df), 2)*100


links_file = sys.argv[1]

file = open(links_file, 'r')
links = file.readlines()
file.close()

links = [link.replace('\n', '') for link in links if len(link) > 0]


tot_neg = 0; tot_pos = 0; tot_sent = 0

print("====== STARTING EXECUTION ======")

for link in links:

    data, _ = extract_content(link, ['dd', 'p', 'li'])
    filterd_data = filter_data(data, 4)
    result = annotate_data(filterd_data)

    topic = link.split(':')[-1]
    datalen = len(result)
    neg_perc = calc_percentage(result, "Negative")
    pos_perc = calc_percentage(result, "Positive")
    
    print("For Topic:", topic)
    print("Total sentences:", datalen)
    print("% of Negative:", neg_perc)
    print("% of Positive:", pos_perc)
    print()

    tot_neg += int(round((neg_perc/100)*datalen, 2))
    tot_pos += int(round((pos_perc/100)*datalen, 2))
    tot_sent += datalen
    

print("\n============================== FINAL RESULT =================================\n")

tot_neg_perc = round((tot_neg/tot_sent)*100, 2)
tot_pos_perc = round((tot_pos/tot_sent)*100, 2)
print("Total % of Negative sentences:", tot_neg_perc)
print("Total % of Positive sentences:", tot_pos_perc)
print("Total % of Neutral sentences:", round(100 - tot_neg_perc - tot_pos_perc, 2))

print("\n============================================================================\n")

