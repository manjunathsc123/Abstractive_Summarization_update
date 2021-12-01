from AbstractiveSummarizer import make_data_to_trainable
from AbstractiveSummarizer import data_cleaning
from AbstractiveSummarizer import data_preprocessing
from AbstractiveSummarizer import train_model
from AbstractiveSummarizer import model_prediction
import argparse
import os
import pandas as pd
import pickle

MAX_SUMMARY_LENGTH = 60
MAX_TEXT_LENGTH=750

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument('--data_csv_path', default=None, help='CSV File Path for Trainable CSV Data')
    parser.add_argument('--data_folder_path', default=None, help='Folder path conaining Stories and Summaries')
    parser.add_argument('--model_folder', default=None, help='Folder path containing model artifacts')
    parser.add_argument('--mode', default='train', help='training or Inference mode')
    parser.add_argument('--split_ratio', default=0.1, type=float, help='Train test split Ratio')
    parser.add_argument('--max_input_length', default=MAX_TEXT_LENGTH, help='Max Input Length')
    parser.add_argument('--max_output_length', default=MAX_SUMMARY_LENGTH, help='Max Output Length')

    args=parser.parse_args()
    data_csv_path = args.data_csv_path
    data_folder_path = args.data_folder_path
    model_folder = args.model_folder
    mode = args.mode
    split_ratio = args.split_ratio
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_file_path = os.path.join(model_folder, 'TextSummarizerModel.pickle')
    # data_csv_path = os.path.join(model_folder, 'train_data.csv')
    if data_csv_path is None:
        data_csv_path = os.path.join(model_folder, 'train_data.csv')
        train_file_list = [os.path.join(data_folder_path, x) for x in os.listdir(data_folder_path)]
        data_df = make_data_to_trainable.run(train_file_list)
        data_df.to_csv(data_csv_path, index=False)
    else:
        data_df = pd.read_csv(data_csv_path)

    print("Cleaning data in Progress...")
    cleaned_text, cleaned_summary = data_cleaning.run(data_df, mode=mode)
    print("Done")
    if mode == 'train':
        print("Train Test Spliting....")
        x_train, x_val, y_train, y_val, x_vocab, y_vocab, x_tokenizer, y_tokenizer = data_preprocessing.run(cleaned_text, cleaned_summary, split_ratio=split_ratio)
        print("Done")

        print("Train started....")
        model, encoder_model, decoder_model = train_model.run(x_train, y_train, x_val, y_val, x_vocab, y_vocab, max_input_length=MAX_TEXT_LENGTH, max_output_length=MAX_SUMMARY_LENGTH)
        print("Done")
        f = open(model_file_path, 'wb')
        pickle.dump([model, encoder_model, decoder_model, x_vocab, y_vocab, x_tokenizer, y_tokenizer], f)
        f.close()
        model_prediction.init(x_tokenizer, y_tokenizer, encoder_model, decoder_model)
        for i in range(len(x_train)):
            print("Actual summary:",model_prediction.seq2summary(y_train[i]))
            this_predicted_summary = model_prediction.run(x_train[i])
            print("Predicted Summary", this_predicted_summary)
    else:
        f = open(model_file_path, 'rb')
        [model, encoder_model, decoder_model, x_vocab, y_vocab, x_tokenizer, y_tokenizer] = pickle.load(f)
        f.close()
        input_token_seq = data_preprocessing.tokenize_text(cleaned_text, x_tokenizer, max_seq_length=MAX_TEXT_LENGTH)
        model_prediction.init(x_tokenizer, y_tokenizer, encoder_model, decoder_model)
        predicted_summary = []
        for i in range(len(input_token_seq)):
            print("Actual Summary", cleaned_summary[i])
            this_predicted_summary = model_prediction.run(input_token_seq[i])
            print("Predicted Summary", this_predicted_summary)
