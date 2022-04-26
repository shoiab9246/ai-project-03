# a3 report
This assignment was about applying different statistical techniques (Naive Bayes, Hidden Markov Models (HMM) and Gibbs Sampling) to topics in domains such as natural language processing (NLP) and image processing.

## Part 1
Part-Of-Speech (POS) tagging

### Problem formulation
Given a corpus of labeled training data with (word, part of speech) pairs, we can build a simple model: P(s|w) using the independence assumption of Naive Bayes. It's important to use smoothing (or pseudocounts) to ensure probabilities of previously-unseen words in the test set have a small (but non-zero) probability.

We can also build a HMM by assuming the previous POS has some influence on the current POS. The Viterbi algorithm essentially states that for the current POS `i`, there are a set of previous POSs with different probabilities. For example the sequence `noun verb` occurs with high probability, but the sequence `verb noun` occurs with a different probability. Many words can have more than one POS. For example "jumps" can be a noun or a verb.

Finally, we can improve by making a more complex model using Markov Chain Monte Carlo (MCMC) method. This is implemented using Gibbs sampling, which resamples each possible POS for each word in the sentence. By repeatedly resampling for many iterations, the resulting particles eventually approach the true distribution of samples.

Therefore we want to use this model to make a better prediction of the POS of each word in the sentence.

### How it works
During training, the code counts:
1. each word
2. each POS for the given word
3. each POS, regardless of what word
4. the frequency of each 2-POS bigram

These counts are then normalized. Items (1,2,3) are divided by the total word count, and 2-POS bigrams are divide by the count of the first POS in their respective sequence.

For computing the Simple model prediction, the most-common POS is chosen for each word. The resulting accuracy is about 90% for words, and 40% for sentences.

For computing the HMM model prediction, the Viterbi algorithm is used to help choose the most likely word, given the previous part of speech for a given word is more likely for some combinations than for others.

For computing the MCMC model prediction, first an arbitrary particle is generated. In our case we just choose each POS to be a noun. Then for the first iteration, each word's POS is resampled using a gaussian distribution.

### Discussion
The hardest part for this problem was correctly formulating the HMM and MCMC models. One design decision was to encode the transition probabilities as a square matrix. Another design decision was to also capture the probability of each POS being the first word in a sentence. This allowed for making a hopefully better guess for the emission probability of the first word's POS.

## Part 2
Mountain Finding
### Problem Formulation
Training images are low-resolution images. Provided code computes an "edge intensity" image using a Sobel filter.
This edge intensity image is our input. The goal is to find the most likely line of pixels which touch the mountain's profile, and highlight those pixels by drawing them in a bright color on the output image.

Like in Part 1, we use models of varying complexity to try to predict the most likely pixels.

### How it works
The Naive Bayes classifier assumes each pixel in each column of the image is independent of the other columns.
For some "easy" mountain images, this classifier can find the mountain top because the largest edge intensity always matches the mountain ridge. 

However, for other images (such as the one with Mount Fuji in the far distance) other features of the image are much brighter in the edge intensity image. One way to improve the average performance of the Naive Bayes classifier is to massage the edge intensity image. For example, in this training image set, the horizontal image center always contains part of the mountain ridge. Also, the mountain is always contained within the top half of the image. So we can use these observations to artificially increase the intensity of the edges in the upper/center portion, thus boosting their probabilities relative to the other edges.

This hack improves the accuracy of the naive Bayes classifier in most of the mountain images.

The next improvement is to use HMM to capure the fact that adjacent columns are not actually independent. If row 50 of the image is part of the mountain, then there's a high probability that row 50 of the adjacent columns is also part of the mountain. There's a low (but nonzero) probability that other rows in adjacent columns are also part of the mountain. 

We can tune HMM with a gaussian distribution such that pixels in the next column which are close to the current row are favored over far-away rows.

Finally, by accepting human input, we assume the human chose correctly and `P(mountain|human_input) = 1`. By again using Viterbi's algorithm, when processing columns from left to right, when we encounter the pixel provided by the human, we're guaranteed to put that rows in the mountain category, and Viterbi will ensure that neighboring columns are very likely to put nearby rows in the mountain category also.

### Discussion
One design decision was to use numpy to process the image. Another design decision was to use heuristics to improve the edge intensity image. This would not work for an arbitrary image (e.g. a rotated image, or an image showing the mountain from above) but it will probably work for most images that people take of mountains.

## Part 3
Optical Character Recognition (OCR)


## Part 3
## Formulation of the problem
For this problem we have used Viterbi algorithm, Bayes nets and Variable elimination. The two main modesl used here are Bayes net and HMM. Bayes Net is used as a classifier to classify the letters fron numbers and puncatuation. The HMM model is used to predict the occurence of characters followed by the previous.For instance the occurence of second character after first,third after the second and so on. We broke the problem into functions which function representing the models such as Bayes net and Hidden Markov Model etc. This basically is the principle followed by OCR(Optical Character Recognition)

## How it Works.
First we create a text file which stores the string of characters of the texts of the images. Now coming to the first function which is calculating probability using Naive Bayes. Firslty a numpy matrix is created Tl and then a variable P which is the probability of the letters from the corpus. Next we can implement a HMM(Hidden Markov Model). Firstly we have to carry out variable elimination. For this we calculate the transistional probabilites and store it in a dictionary. Then we caclulate the tau values of each characters and the sum. The variables are elminated based on the sum.Viterbi can be used to calculate the most likely path through different obsevations.

For computing letter[i] we have to find the prodcut of transitional probabilites along the path and the probabilites of obsevation given at a particular state.(This is based on the reference given in the textbook by Russell and Norvig the p.578.

Next coming to the Viterbi algorithm, the viterbi algorithm works with the logic that the first we calculate the emission probabilities. It then calculates the most likely sequence of the hidden states. It backtracks until the maximum value or the best solution is obtained and it returns that string.

The probability of a given letters depends on the following :
1)Guessing the current letter
2)emssion probability
3)transition probability

Here have two variables state and observation where state is the predicted letter from the next and the observation is the guess for the current letter. 


## Challenges Faced
We have implemented the Naives Bayes classifier and its working without any hassle.However we faced issues working on the HMM model. We had tried several approaches. One  of the approaches was throwing an error which occured due to indexing issues. This was one of the main challenges faced.
