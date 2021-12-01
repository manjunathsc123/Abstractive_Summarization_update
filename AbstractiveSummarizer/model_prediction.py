import numpy as np

MAX_TEXT_LENGTH = 750
MAX_SUMMARY_LENGTH = 60

def init(x_tokenizer, y_tokenizer, model_encoder, model_decoder):
    global  reverse_target_word_index, reverse_source_word_index, target_word_index, encoder_model, decoder_model
    encoder_model = model_encoder
    decoder_model = model_decoder
    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index

def predict(input_seq, max_output_length=MAX_SUMMARY_LENGTH):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='eostok'):
            decoded_sentence += ' '+ sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_output_length-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    output_text = ''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            output_text = output_text + reverse_target_word_index[i] + ' '
    return output_text

def seq2text(input_seq):
    output_text = ''
    for i in input_seq:
        if i!=0:
            output_text = output_text + reverse_source_word_index[i]+' '
    return output_text

def run(input_x):
    return predict(input_x.reshape(1,MAX_TEXT_LENGTH))
