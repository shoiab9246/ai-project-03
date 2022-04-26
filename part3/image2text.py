#!/usr/bin/python
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors:
# Alex Shroyer, spanampi, Shoiab Mohammed
#
# (based on skeleton code by D. Crandall, Oct 2020)
from PIL import Image # , ImageDraw, ImageFont
import numpy as np
import sys
CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

def print_letter(letter):
    print("\n".join(list("".join(list(p)) for p in np.where(letter, '*', ' '))))

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    col, row = im.size
    col = col // CHARACTER_WIDTH * CHARACTER_WIDTH # truncate extra pixels
    return np.transpose(
        (255 > (np.array([px[i,j] for i in range(col) for j in range(row)]))
         ).astype(int).reshape(col//CHARACTER_WIDTH, CHARACTER_WIDTH, CHARACTER_HEIGHT),
        axes=(0,2,1))

def ltr():
    # since we're not allowed ot use globals...
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

def load_training_letters(fname):
    letter_images = load_letters(fname)
    return {ltr()[i]: letter_images[i] for i in range(len(ltr()))}

def read_data(fname):
    exemplars = []
    with open(fname) as f:
        for line in f:
            exemplars += [w for w in line.split()][::2]
    return exemplars

def probs(corpus, tl):
    '''
    T: letter[i]->letter[i+1] probability matrix from text corpus
    P: probability of each letter
    '''
    T = np.ones((tl.size, tl.size))
    P = {i:0 for i in tl}
    TI = {k:i for i,k in enumerate(ltr())} # faster indexing
    last_letter = None
    total_letters = 0
    for word in corpus:
        for l in word:
            if l in P:
                if last_letter is not None:
                    T[TI[l], TI[last_letter]] += 1
                last_letter = l
                P[l] += 1
            total_letters += 1
    P = {k:v/total_letters for k,v in P.items()}
    T /= T.sum(axis=0)
    return P,T


def main(args=None):
    itr, txt, ite = (("courier-train.png", "../part1/bc.train", "test_images/test-6-0.png")
                     if args is None else sys.argv[1:])
    tl = np.array(list(ltr()))
    train_letters = load_training_letters(itr)
    test_letters = load_letters(ite)

    # naive bayes solution
    nb = [np.argmax([np.sum(letter == train_letters[i]) for i in train_letters]) for letter in test_letters]
    print("Simple: " + ''.join(tl[nb]))

    # TODO HMM solution
    # Probabilty of the given letter depends on:
    # 1. the guesses for what the current letter could be (based on the image)
    # 2. the probability of each of those letters overall (emission probablity)
    # 3. the probability of letter[i] given letter[i-1] (transition probability)
    #
    # To compute letter[i], we need to find
    # "the product of the transition probabilities along the path and the
    # probabilities of the given observations at each state" -- Russel & Norvig p.578
    #
    # For this problem, I think that means p(state[i]) = p(state[i-1] -> state[i]) * p(observation[i])
    # where "state" is the predicted letter from the training text,
    # and "observation" is the guess for the current letter.
    #
    # The tricky part is that "observation" has multiple letters with different probability.
    # Viterbi can be used to choose the most likely path through those diffferent obervations.

    txt = read_data(txt)
    P, T = probs(txt, tl)
    # print(T)
    # print("   HMM: " + ''.join(tl[hmm])) ## TODO

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")
    main(sys.argv[1:])
