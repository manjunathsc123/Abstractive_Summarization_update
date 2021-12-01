import os
import re
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('punkt')


def read_text_file(text_file_path):
    # print('Reading file in ', text_file_path)
    with open(text_file_path, 'r',encoding = "utf8") as f:
        text_data = f.read()
    return text_data

def save_text_file(text_data, output_file_path):
    # print('Saving file in ', output_file_path)
    with open(output_file_path, 'w') as f:
        f.write(text_data)

def read_ground_summary(gt_text):
    pattern = re.compile(r'@highlight')
    highlight_index_list = []
    for this_match in pattern.finditer(gt_text):
        highlight_index_list.append([this_match.start(), this_match.end()])
    if not len(highlight_index_list):
        return gt_text, ''
    full_text = gt_text[0:highlight_index_list[0][0]]
    summary_text = []
    for i in range(0, len(highlight_index_list)-1):
        summary_text.append(gt_text[highlight_index_list[i][1]:highlight_index_list[i+1][0]])
    summary_text.append(gt_text[highlight_index_list[-1][1]:])
    summary_text = '\n'.join(summary_text)
    return full_text, summary_text

def run(train_file_list):
    train_df = pd.DataFrame()
    for this_file_path in tqdm(train_file_list):
        input_text = read_text_file(this_file_path)
        this_file_actual_text, this_summary = read_ground_summary(input_text)
        train_df = train_df.append([[this_file_actual_text, this_summary]])
    train_df.reset_index(drop=True, inplace=True)
    train_df.columns = ['InputText', 'Summary']
    return train_df

