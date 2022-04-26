# First row of P is an array of pdfs of music values,
# second row is an array of pdfs of applause values.
# Some people started applauding too early, so the goal
# is to detect when the applause should really start.
import matplotlib.pyplot as plt
import numpy as np

# viterbi algorithm
# print(P.shape) # (2, 962)
# iterate through forward
T = np.array([[0.9, 0.1], [0, 1]]) # transition probabilities
pbar = P.copy() # really just care about initializing the first column values (initial probability)
B = np.empty(pbar.shape)
for n in range(2): # for each row
  for i in range(B.shape[1]-1): # for each column except last
    b = np.argmax(T[:,n] * pbar[:,i]) # did music (0) or clap (1) have the higher transition prob?
    B[n,i] = b
    pbar[n,i+1] = np.prod([T[b,n], P[b,i], P[n,i+1]]) # compute next sample probability

pbar /= np.sum(pbar, axis=0) # normalize so they're still probabilities {0..1}

# plt.figure(figsize=(12,6))
# plt.imshow(pbar, aspect='auto')
# plt.ylim(-0.1,1.1)
# plt.title(r'$\bar{P}$, normalized')

# iterate backward, find most likely path through graph
state = np.zeros(pbar.shape[1], dtype=int)
state[-1] = np.argmax(pbar[:,-1])
for i in range(i,0,-1):
  state[i] = B[state[i+1],i]

sq = np.ma.masked_where(state<0.5, state)
sc = np.ma.masked_where(state>0.5, state)

# plt.plot(sq, color='r', label='time to clap')
# plt.plot(sc, color='b', label='quiet please')
# plt.title('when to clap?')
# plt.legend()
# plt.show()
