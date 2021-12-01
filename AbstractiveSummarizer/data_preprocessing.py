import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences


MAX_SUMMARY_LENGTH = 60
MAX_TEXT_LENGTH=750


def filter_data(text_data: list, summary_data: list):
    short_text=[]
    short_headlines=[]
    for i in range(len(text_data)):
        if(len(summary_data[i].split())<=MAX_SUMMARY_LENGTH and len(text_data[i].split())<=MAX_TEXT_LENGTH):
            short_text.append(text_data[i])
            short_headlines.append(summary_data[i])
    data_df = pd.DataFrame({'text':short_text,'summary':short_headlines})
    data_df['summary'] = data_df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')
    return data_df

def get_train_val_data(data_df, split_ratio=0.1):
    x_train, x_val, y_train, y_val = train_test_split(np.array(data_df['text']),np.array(data_df['summary']),test_size=split_ratio,random_state=0,shuffle=True)
    return x_train, x_val, y_train, y_val

def tokenize_data(train_data, val_data, max_seq_length=MAX_TEXT_LENGTH):
    data_tokenizer = Tokenizer() 
    data_tokenizer.fit_on_texts(list(train_data))

    thresh=4
    cnt=0
    tot_cnt=0
    freq=0
    tot_freq=0

    for key,value in data_tokenizer.word_counts.items():
        tot_cnt = tot_cnt+1
        tot_freq = tot_freq+value
        if(value<thresh):
            cnt = cnt+1
            freq = freq + value

    print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
    print("Total Coverage of rare words:",(freq/tot_freq)*100)

    #prepare a tokenizer for reviews on training data
    data_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
    data_tokenizer.fit_on_texts(list(train_data))

    train_data = tokenize_text(train_data, data_tokenizer, max_seq_length, padding='post')
    val_data = tokenize_text(val_data, data_tokenizer, max_seq_length, padding='post')

    #size of vocabulary ( +1 for padding token)
    data_vocab = data_tokenizer.num_words + 1
    return train_data, val_data, data_vocab, data_tokenizer

def tokenize_text(input_data, data_tokenizer, max_seq_length, padding='post'):
    #convert text sequences into integer sequences
    data_seq = data_tokenizer.texts_to_sequences(input_data)
    #padding zero upto maximum length
    return pad_sequences(data_seq,  maxlen=max_seq_length, padding=padding)

def delete_unwanted(x_data, y_data):
    ind=[]
    for i in range(len(y_data)):
        cnt=0
        for j in y_data[i]:
            if j!=0:
                cnt=cnt+1
        if(cnt==2):
            ind.append(i)
    y_data=np.delete(y_data,ind, axis=0)
    x_data=np.delete(x_data,ind, axis=0)
    return x_data, y_data

def run(text_data: list, summary_data: list, split_ratio=0.1):
    filtered_data_df = filter_data(text_data, summary_data)
    x_train, x_val, y_train, y_val = get_train_val_data(filtered_data_df, split_ratio=split_ratio)
    x_train, x_val, x_vocab, x_tokenizer = tokenize_data(x_train, x_val, max_seq_length=MAX_TEXT_LENGTH)
    y_train, y_val, y_vocab, y_tokenizer = tokenize_data(y_train, y_val, max_seq_length=MAX_SUMMARY_LENGTH)
    x_train, y_train = delete_unwanted(x_train, y_train)
    x_val, y_val = delete_unwanted(x_val, y_val)
    return x_train, x_val, y_train, y_val, x_vocab, y_vocab, x_tokenizer, y_tokenizer
