from __future__ import unicode_literals, print_function, division
from util.DataConversionUtil import DataConversionUtil
from util.LanguageUtil import prepareData, tensorsFromPair, tensorFromSentence, prepareValData
import random
import torch
import torch.nn as nn
from torch import optim
from util.graph_plotter import showPlot
from baseline.model.encoder import EncoderRNN
from baseline.model.decoder import DecoderRNN, AttnDecoderRNN
from util.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Dynamically allocate encoder_outputs based on input_length
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0

    for ei in range(input_length):  
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]  # Store encoder output

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True  

    if use_teacher_forcing:
        for di in range(target_length):  
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[:input_length]
            )  
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[:input_length]
            )  
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def hyperparam(hidden_size):
    global input_lang, output_lang, pairs
    input_lang, output_lang, pairs = prepareValData("en", "sql")
    
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    
    lr_candidates = [0.0001, 0.001, 0.01, 0.1, 1]
    best_lr = lr_candidates[0]
    highest_accuracy = 0
    
    for lr in lr_candidates:
        print(f"Testing learning rate: {lr}")
        trainIters(encoder1, attn_decoder1, 1000, print_every=500, plot_every=500, learning_rate=lr)
        accuracy = evaluateRandomly(encoder1, attn_decoder1, n=10)
        
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_lr = lr
    
    return best_lr


def trainIters(encoder, decoder, n_iters, print_every=10, plot_every=20, learning_rate=0.0005):
    plot_losses = []
    print_loss_total = 0  
    plot_loss_total = 0  

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang) for _ in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        input_tensor, target_tensor = training_pairs[iter - 1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'({iter} {iter / n_iters * 100:.2f}%) {print_loss_avg:.4f}')

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses, "Baseline loss")


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        seq_len = input_tensor.size(0)  # Get actual input sequence length
        encoder_outputs = torch.zeros(seq_len, encoder.hidden_size, device=device)  # Dynamically adjust size

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  
        decoder_hidden = encoder_hidden

        decoded_words = []
        seq_len = input_tensor.size(0)  # Get actual input length
        decoder_attentions = torch.zeros(seq_len, seq_len)  # Adjust to sequence length


        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[:input_length]
            )  
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=1000):
    correct = 0
    for _ in range(n):
        pair = random.choice(pairs)
        print(f'\nEnglish Question: {pair[0]}')
        print(f'Ground truth Query: {pair[1]}')

        generated_tokens, _ = evaluate(encoder, decoder, pair[0])
        generated_query = ' '.join(generated_tokens)

        if generated_query.strip() == pair[1].strip():
            correct += 1

        print(f'Generated Query: {generated_query}')
    
    accuracy = (correct / n) * 100
    print(f"\n\nCorrect Examples: {correct} out of {n}")
    return accuracy


def run_baseline():
    hidden_size = 256
    x = DataConversionUtil()
    
    best_lr = hyperparam(hidden_size)
    
    global input_lang, output_lang, pairs
    input_lang, output_lang, pairs = prepareData("en", "sql")
    
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    
    trainIters(encoder1, attn_decoder1, 250000, print_every=1000, plot_every=1000, learning_rate=best_lr)
    
    accuracy = evaluateRandomly(encoder1, attn_decoder1, n=1000)
    print(f"Final Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    run_baseline()
    print("Baseline model completed")