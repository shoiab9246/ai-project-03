###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
# Alex Shroyer , Shoiab Mohammed
#
# (Based on skeleton code by D. Crandall)
import numpy as np
import matplotlib.pyplot as plt

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
class Solver:
    def __init__(self):
        self.alpha = 1e-9 # used in pseudocount for unseen word
        self.counts = {} # {word: int}
        self.words = {} # {word: {pos: int}}
        self.pos_keys = ['det', 'noun', 'adj', 'verb', 'adp', '.',
                         'adv', 'conj', 'prt', 'pron', 'num', 'x']
        self.pk = {k:i for i,k in enumerate(self.pos_keys)} # for simpler lookup
        self.pk.update({i:k for i,k in enumerate(self.pos_keys)})
        # Transition Probability matrix for POS: [row,col] is [pos[i-1], pos[i]]
        self.tp = np.ones((len(self.pos_keys), len(self.pos_keys)))
        self.ppos= {k:0 for k in self.pos_keys} # count of each pos (anywhere in sentence)
        self.first = {k:0 for k in self.pos_keys} # count of pos when it is the first word
        self.first_most_likely = ''
        self.totalwords = 0 # total for entire dataset (same as count of pos)

    # Calculate the log of the posterior probability of a given sentence
    # with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        # print(label)
        if model == "Simple":
            # If we assume each word is independent, e.g. P(i,j) = P(i)P(j), then
            # P(x) is probability x occurs in dataset: (count x) / (count words)
            probs = np.array([
                self.ppos[s] * self.counts.get(w, self.alpha)
                for w,s in zip(sentence, label)
            ])
            return sum(np.log(probs / self.totalwords))
        elif model == "HMM":
            return -999
        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    # This method must be called before self.posterior
    def train(self, data):
        # build posterior histograms for word:parts and part:words

        for d in data:
            self.first[d[1][0]] += 1 # first pos in sentence

            prev_pos = None
            for w,c in zip(*d): # (word, class)
                self.totalwords += 1
                self.ppos[c] += 1
                self.counts[w] = self.counts.get(w, 0) + 1

                # update adjacency matrix (pos[i-1] -> pos[i])
                if prev_pos is not None:
                    self.tp[self.pk[prev_pos], self.pk[c]] += 1
                prev_pos = c

                # update part of speech count for the given word
                if w in self.words:
                    self.words[w][c] = self.words[w].get(c, 0) + 1
                else:
                    self.words[w] = {c:1}

        # normalize self.tp by first word
        self.tp /= self.tp.sum(axis=0)

        # TODO normalize firstword
        print(self.first)
        sum_first = sum(self.first.values())
        self.first = {k:v/sum_first for k,v in self.first.items()}
        # for x in self.first:
        #     self.first[x] /= sum(self.first.values())

        # normalize ppos
        for x in self.ppos:
            self.ppos[x] /= self.totalwords

        # normalize word[w][c]
        for x in self.words:
            for y in self.words[x]:
                self.words[x][y] /= (self.totalwords)

        self.first_most_likely = sorted(self.first.items(), key=lambda x:x[1])[-1]

    # For a given sentence, use posterior probabilities to return a list of parts of speech for the sentence.
    def simplified(self, sentence):
        parts = []
        most_common_pos = sorted(self.ppos.items(), key=lambda x:x[1])[-1]
        for w in sentence:
            if w in self.words:
                poss, counts = list(self.words[w].keys()), list(self.words[w].values())
                parts.append(poss[np.argmax(counts)])
            else:
                parts.append(most_common_pos) # use most common POS if we never saw this word before

        return parts

    def hmm_viterbi(self, sentence):
        # columns of P are POS, rows are words from sentence
        P = np.zeros((12, len(sentence)))
        for i,w in enumerate(sentence):
            if w in self.words:
                for c in self.words[w]: # it can be multiple POS classes
                    P[self.pk[c],i] = self.words[w][c]
            else:
                P[:,i] = self.alpha

        print(P)


        # pbar = P.copy() # we'll replace all but first column
        # s0 = sentence[0] # first word
        # if s0 in self.words:
        #     for c in self.words[s0]:
        #         pbar[self.pk[c],0] = self.words[s0][c] * self.first[c]
        # else:
        #     pbar[:,0] = self.alpha

        # print(pbar)

        # B = np.zeros(pbar.shape, dtype=int) # indexes

        # first word is special since there's no "previous word"

        # # forward pass: P(wi|cj) get probs for each pos for each word in sentence
        # for n in range(12): # for each pos (row)
        #     # P(s2|s1) = self.tp[i, i+1]
        #     for i in range(B.shape[1]-1): # for each word (column)
        #         if i==0:
        #             firsts = np.log(np.array(list(self.first.values())) + self.alpha)
        #             pbarln = np.log(pbar[:,i] + self.alpha)
        #             tplog = np.log(self.tp[0,1] + self.alpha)
        #             b = np.argmax([firsts + pbarln + tplog])
        #         else:
        #             b = np.argmax(self.ppos[self.pk[n]] * pbar[:,i]) # class with highest transition prob
        #         B[n,i] = b
        #         pbar[n,i+1] = self.tp[b,n] * P[b,i] * P[n,i+1] # compute next sample probability

        # print(pbar)
        # # iterate backward, find most likely path through graph
        # state = np.zeros(pbar.shape[1], dtype=int)
        # state[-1] = np.argmax(pbar[:,-1])
        # for i in range(B.shape[1]-2, -1, -1):
        #     state[i] = B[state[i+1],i]

        # # print(state)
        # return list(np.array(self.pos_keys)[state])


    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the
    # interface the same, but you can change the code itself.  It should return
    # a list of part-of-speech labelings of the sentence, one part of speech per
    # word.
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

# rule for factoring:
# look @ every var;
# need a term for each var, conditioned on its parents
#
# Emission prob: P(w1|s1) e.g. P(word=dog|state=noun)
# or for text, P(image|letter)
#
# P(HMM) = P(s1) P(s2|s1)... P(w1|s1)...
#
#
# To compute s1, design choice. Can just look at training corpus and
# find most probable 1st word.
#
# Mountain finding:
# make human-specified point have emission probability 1.
#
# If using gaussian for 2.3, if you put in log space, gaussian becomes intuitiive:
# P(sj|si) = P'(sj-si)
# ln(x) = +(sj - si)^2 / sigma^2
#
# MCMC - P(s1) * P(w1=jumps|s1) * P(s2=N|s1=N)
# There are many different valid MCMC formulations.
# The main one is: for each particle, resample each variable,
# in sequence, until all have been resampled.
# Other formulations are also valid, such as resample just 1 var per particle,
# or resample in random order, or resample some (not all).
#
# Spaces!
# they don't occur next to each other
# s->s is very rare
# But spaces are very common in general
# and a noisy space is similar to the average letter
# Also maybe black->white has different transition prob than white->black
