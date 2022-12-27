#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
import math


CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
TRAIN_letter_len = len(TRAIN_LETTERS)



def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result
    

def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


# Reads the training.txt file and returns "first letter of words", "all letters of the file" and "number of occurences of all letters"
def read_train_file(fname):
    train_file = open(fname,'r')                                
    train_lines = train_file.readlines()

    # Stores first letter of all words onto a list
    first_letter_of_word =[]
    [first_letter_of_word.append(word[0]) for line in train_lines for word in line.split()]

    # Stores all letters onto list
    all_letters = []
    [all_letters.append(letter) for line in train_lines for letter in line]

    # Stores occurence of all letters onto list
    letter_occurence = np.zeros(TRAIN_letter_len)
    letter_occurence = [sum(1 for l in all_letters if t_letter==l) for t_letter in TRAIN_LETTERS]

    train_file.close()

    return first_letter_of_word, all_letters, letter_occurence

# Calculates Initial Probability
def initial_prob(first_letter_of_word):

    # Stores number of occurences of each character
    init_prob = np.zeros(TRAIN_letter_len) 
    init_prob = [sum(1 for w in first_letter_of_word if t_letter==w) for t_letter in TRAIN_LETTERS]

    # Laplace Smoothing
    # Formula taken from - "https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece"
    alpha= 1
    init_prob = [(i+alpha) / ((len(first_letter_of_word) + (2*alpha))) for i in init_prob]

    return init_prob

# Calculates Transition Probability
def transition_prob(all_letters , letter_occurence):

    # Stores Transition Occurences
    tran_prob = np.zeros(shape=(TRAIN_letter_len, TRAIN_letter_len))
    for i in range(0, len(all_letters)-1):
        if all_letters[i] in TRAIN_LETTERS and all_letters[i+1] in TRAIN_LETTERS:
            tran_prob[TRAIN_LETTERS.index(all_letters[i]),TRAIN_LETTERS.index(all_letters[i+1])] += 1

    # Laplace Smoothing
    alpha = 1
    for i in range (0,TRAIN_letter_len):
        for j in range (0,TRAIN_letter_len):
            tran_prob[i][j] = (tran_prob[i][j]+alpha) / (letter_occurence[i]+(2*alpha))

    return tran_prob

# Calculates total number of black pixels or "*" in training and test files
def calc_total_black(train_letters, test_letters):
    
    total_black_train=0
    for i in train_letters:
        letter=train_letters.get(i)
        for pixel_line in letter:
            for pixel in pixel_line:
                if pixel == '*':
                    total_black_train += 1
    
    total_black_test=0
    for letter in test_letters:
        for pixel_line in letter:
            for pixel in pixel_line:
                if pixel == '*':
                    total_black_test += 1
    
    return total_black_train, total_black_test

# Calculates Emission Probability
def emission_prob(train_letters, test_letters):
    
    emis_prob= np.zeros(shape=(len(train_letters),len(test_letters)))
    
    # We calculate the density of pixels for training and testing datasets
    total_black_train, total_black_test = calc_total_black(train_letters, test_letters)
    black_train_density = total_black_train/len(test_letters)
    black_test_density =  total_black_test/len(train_letters)

    for train_l in train_letters:
        for test_l in range(len(test_letters)):
            actual = train_letters.get(train_l)
            observed = test_letters[test_l]
            
            # We check for matching black and white as well as mismatch black and white pixels.
            match_black, match_white, mis_black, mis_white = 0, 0, 0, 0
            for pixel_row in range(CHARACTER_HEIGHT):
                for pixel_col in range(CHARACTER_WIDTH):
                    if actual[pixel_row][pixel_col] == observed[pixel_row][pixel_col]:
                        if observed[pixel_row][pixel_col]=='*':
                            match_black += 1
                        elif observed[pixel_row][pixel_col]==' ':
                            match_white += 1
                    elif actual[pixel_row][pixel_col] == '*':
                        mis_black += 1
                    elif actual[pixel_row][pixel_col] == ' ':
                        mis_white += 1
                
                if (black_test_density) > (black_train_density):    # Different priority is given depeding on pixel densities of train and test files
                    emis_prob[TRAIN_LETTERS.index(train_l)][test_l] = math.pow(0.75, match_black) * math.pow(0.7,match_white) * math.pow(0.25, mis_black) * math.pow(0.3, mis_white)
                else:
                    emis_prob[TRAIN_LETTERS.index(train_l)][test_l] = math.pow(0.95, match_black) * math.pow(0.65,match_white) * math.pow(0.4, mis_black) * math.pow(0.05, mis_white)

    return emis_prob

# Calculates Simple OCR
def simple_ocr(test_letters, emis_prob):

    simplified_prob = np.zeros(len(test_letters))
    simplified_prob = np.argmax(emis_prob, axis=0)          # Returns rows with max probability. The use of "arg_max" was suggested by a fellow classmate ,Sravani Wayangankar.
    
    return simplified_prob


# Backtracks and stores the path with maximum probability
def backtracking(t, best_prob, best_state):
    most_prob_path = np.zeros(t, dtype=int)
    most_prob_path[-1] = np.argmax(best_prob[:,-1])
    
    for i in range(1,t)[::-1]:
        most_prob_path[i-1] = best_state[most_prob_path [i],i]
    return most_prob_path 


# Below code for Viterbi was inspired by Prof.David Crandall's Equation given in PPTs and methodology was inspired from "https://github.com/snehalvartak/Optical-Character-Recognition-using-HMM"
# Calculates Viterbi for Character Recognition
def viterbi_ocr(test_letters, init, tran, emis):
    observed = test_letters
    
    best_prob = np.zeros(shape=(len(train_letters),len(observed)))                              # 2D array stores maximum probability 
    best_state =  np.empty(shape=(len(train_letters), len(observed)), dtype=int)                # Stores the location of most probable states
    
    for i in range(len(train_letters)):                                                         # We find initial probability at t=0 for each state i calculated as initial(i)*emission(time ='t'|i)
        best_prob[i,0] = math.log(init[i]) + math.log(emis[i,0])
    
    for x in range(1,len(observed)):                                                            # We find the probabilites for each state i ending at time t+1
        for y in range(len(train_letters)):
            intermediate_prob = []
            for z in range(len(train_letters)):
                intermediate_prob.append(best_prob[z,x-1] + math.log(tran[z,y]) + math.log(emis[y,x]))  # We use log addition instead of simple multiplication
            best_prob[y,x] = max(intermediate_prob)                                             # We get max probability and store it in our 2D array

            best_state[y,x] = intermediate_prob.index(max(intermediate_prob))
   
    # Backtrack path to find the maximum probabilities
    path = backtracking(len(observed), best_prob, best_state)
    return path



# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

first_letter_of_word, all_letters, letter_occurence = read_train_file(train_txt_fname)

# Calculates Initial, Transtition and Emission probability
init_prob = initial_prob(first_letter_of_word)
tran_prob = transition_prob(all_letters, letter_occurence)
emis_prob = emission_prob(train_letters, test_letters)

# Simplified text is calculated and returned
simple = simple_ocr(test_letters, emis_prob)
simple_text = "".join([TRAIN_LETTERS[i] for i in simple])
print ("Simple: ", simple_text)

# Viterbi is calculated and returned
viterbi = viterbi_ocr(test_letters, init_prob, tran_prob, emis_prob)
viterbi_text = "".join([TRAIN_LETTERS[i] for i in viterbi])
print ("HMM MAP: " , viterbi_text)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
####print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
##print("\n".join([ r for r in test_letters[0] ]))



# The final two train_lines of your output should look something like this:
##print("Simple: " + "Sample s1mple resu1t")
##print("   HMM: " + "Sample simple result") 


