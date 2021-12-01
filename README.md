# AbstractiveTextSummarization
This Repo shall contain training and inference code for Text Abstractive Summarization

# For Training:

Please run the following command for training model
python main.py --data_folder_path='./../data/train' --model_folder='./../model' --mode='train' --split_ratio=0.1

                                            or

python main.py --data_csv_path='./../model/train_data.csv' --model_folder='./../model' --mode='train' --split_ratio=0.1

# For Inference:

Please run the following command
python main.py --data_folder_path='./../data/test' --model_folder='./../model' --mode='test'
