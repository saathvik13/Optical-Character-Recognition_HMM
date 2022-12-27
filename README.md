
## Problem Statement - 
The problem is to show the versatility of HMM in the application of Optical Character Recognition by recognising and predicting characters from an image containing text.  There are 2 algorithms used- 
1) Simply Naive Bayes
2) HMM with MAP inference (VITERBI)


## Algorithm-

1) Read the training file and map to training characters with training image using pixels "*" for black pixel and " " for white pixel with character_width=15 and character_height=14. 

2) Read the training file and map it the same way as training characters.

3) Calculate Initial probability for the training file as,
    
    Initial Probability : number of times a word begins with a particular character / total number of words.

4) Calculate Transtition probability for the training file as,
    
    Transtition Probability : number of times a letter 'i' is followed by another character 'i+1' / total number of initial character 'i'.

5) Perform Laplace smoothing for Initial and Transtition probabilites since few value counts canbe NULL and performing division and multiplication operations while using the VITERBI function can result in 0 and there are chances that this character is never picked.

Source: "https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece"

5) Calculate Emission probability for the training file as, the probability of a given character actually being that character
    
    Emission Probability : number of times a letter 'i' is followed by another letter 'i+1' / total number of initial character'i'.
    
    a) First, This is calculated by iterating through the each character in the training file and iterating through each character in testing file and then find the number of matching black and white pixels and number of mismatching balck and white pixels for each of the characters.
    
    b) Next, build a table with the probabilites with training_letters as rows and test_letters as columns with cell values being the probabilities using the formula which is determined by the black pixel density and white density.
    
    c) Since there is a lot of noise, ie- a case where the density of test black pixel is more that train black pixels, we give a higher weightage to the black pixels considering the noise. 
        
        Case 1: (black_test_density > black_train_density) = 0.75^black_match * 0.7^white_match * 0.25^black_mismatch * 0.3^white_mismatch
        
        Case 2:  (black_test_density > black_train_density) = 0.95^black_match * 0.65^white_match * 0.4^black_mismatch * 0.0.5^white_mismatch
    
    These values were determined by training the data over and over again.

6) Perform Simply Naive Bayes using emission probabilites already calculated. The 'argmax' function is used which automatically picks the rows with highest probabilities. Therefore, maximum probability of each character calculated using pixel match and mismatch is outputted.

7) Perform MAP HMM (Viterbi) which used initial, transition and emission probabilities. Viterbi is calculated using : P(Q0=q0) *(product {t=0 to T} P(Qt+1|Q_t)) * (product_{t=0 to T} P(Observed(t)|Q(t))


## Problems Faced-

Finding the right values for emission was a hard task. Initially, equal priority was given to black, white for matching and mismatching pixels and then it was tweaked till a good accuracy was obtained but it was still not satisfactory. So, pixel density for test and train was calculated seperately and different values of emission is given for both the cases above.