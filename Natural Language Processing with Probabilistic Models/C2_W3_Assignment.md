# Assignment 3: Language Models: Auto-Complete

In this assignment, you will build an auto-complete system.  Auto-complete system is something you may see every day
- When you google something, you often have suggestions to help you complete your search. 
- When you are writing an email, you get suggestions telling you possible endings to your sentence.  

By the end of this assignment, you will develop a prototype of such a system.

<img src = "./images/stanford.png" style="width:700px;height:300px;"/>

## Important Note on Submission to the AutoGrader

Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:

1. You have not added any _extra_ `print` statement(s) in the assignment.
2. You have not added any _extra_ code cell(s) in the assignment.
3. You have not changed any of the function parameters.
4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
5. You are not changing the assignment code where it is not required, like creating _extra_ variables.

If you do any of the following, you will get something like, `Grader Error: Grader feedback not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/probabilistic-models-in-nlp/supplement/saGQf/how-to-refresh-your-workspace).
## Table of Contents
- [1 - Load and Preprocess Data](#1)
    - [1.1 - Load the Data](#1.1)
    - [1.2 - Pre-process the Data](#1.2)
        - [Exercise 1- split_to_sentences (UNQ_C1)](#ex-1)
        - [Exercise 2 - tokenize_sentences (UNQ_C2)](#ex-2)
        - [Exercise 3 - get_tokenized_data (UNQ_C3)](#ex-3)
        - [Exercise 4 - count_words (UNQ_C4)](#ex-4)
        - [Exercise 5 - get_words_with_nplus_frequency (UNQ_C5)](#ex-5)
        - [Exercise 6 - replace_oov_words_by_unk (UNQ_C6)](#ex-6)
        - [Exercise 7 - preprocess_data (UNQ_C7)](#ex-7)
- [2 - Develop n-gram based Language Models](#2)
    - [Exercise 8 - count_n_grams (UNQ_C8)](#ex-8)
    - [Exercise 9 - estimate_probability (UNQ_C9)](#ex-9)    
- [3 - Perplexity](#3)
    - [Exercise 10 - calculate_perplexity (UNQ_C10)](#ex-10)
- [4 - Build an Auto-complete System](#4)
    - [Exercise 11 - suggest_a_word (UNQ_C11)](#ex-11)

A key building block for an auto-complete system is a language model.
A language model assigns the probability to a sequence of words, in a way that more "likely" sequences receive higher scores.  For example, 
>"I have a pen" 
is expected to have a higher probability than 
>"I am a pen"
since the first one seems to be a more natural sentence in the real world.

You can take advantage of this probability calculation to develop an auto-complete system.  
Suppose the user typed 
>"I eat scrambled"
Then you can find a word `x`  such that "I eat scrambled x" receives the highest probability.  If x = "eggs", the sentence would be
>"I eat scrambled eggs"

While a variety of language models have been developed, this assignment uses **N-grams**, a simple but powerful method for language modeling.
- N-grams are also used in machine translation and speech recognition. 


Here are the steps of this assignment:

1. Load and preprocess data
    - Load and tokenize data.
    - Split the sentences into train and test sets.
    - Replace words with a low frequency by an unknown marker `<unk>`.
1. Develop N-gram based language models
    - Compute the count of n-grams from a given data set.
    - Estimate the conditional probability of a next word with k-smoothing.
1. Evaluate the N-gram models by computing the perplexity score.
1. Use your own model to suggest an upcoming word given your sentence. 


```python
import math
import random
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')

import w3_unittest
nltk.data.path.append('.')
```

<a name='1'></a>
## 1 - Load and Preprocess Data

<a name='1.1'></a>
### 1.1 - Load the Data
You will use twitter data.
Load the data and view the first few sentences by running the next cell.

Notice that data is a long string that contains many many tweets.
Observe that there is a line break "\n" between tweets.


```python
with open("./data/en_US.twitter.txt", "r") as f:
    data = f.read()
print("Data type:", type(data))
print("Number of letters:", len(data))
print("First 300 letters of the data")
print("-------")
display(data[0:300])
print("-------")

print("Last 300 letters of the data")
print("-------")
display(data[-300:])
print("-------")
```

<a name='1.2'></a>
### 1.2 - Pre-process the Data

Preprocess this data with the following steps:

1. Split data into sentences using "\n" as the delimiter.
1. Split each sentence into tokens. Note that in this assignment we use "token" and "words" interchangeably.
1. Assign sentences into train or test sets.
1. Find tokens that appear at least N times in the training data.
1. Replace tokens that appear less than N times by `<unk>`


Note: we omit validation data in this exercise.
- In real applications, we should hold a part of data as a validation set and use it to tune our training.
- We skip this process for simplicity.

<a name='ex-1'></a>
### Exercise 1- split_to_sentences

Split data into sentences.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li> Use <a href="https://docs.python.org/3/library/stdtypes.html?highlight=split#str.split" >str.split</a> </li>
</ul>
</p>


```python
import math
import random
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')

import w3_unittest
nltk.data.path.append('.')
with open("./data/en_US.twitter.txt", "r") as f:
    data = f.read()
print("Data type:", type(data))
print("Number of letters:", len(data))
print("First 300 letters of the data")
print("-------")
display(data[0:300])
print("-------")
print("Last 300 letters of the data")
print("-------")
display(data[-300:])
print("-------")
def split_to_sentences(data):
    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences   
```

    [nltk_data] Downloading package punkt to /home/jovyan/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.


    Data type: <class 'str'>
    Number of letters: 3335477
    First 300 letters of the data
    -------



    "How are you? Btw thanks for the RT. You gonna be in DC anytime soon? Love to see you. Been way, way too long.\nWhen you meet someone special... you'll know. Your heart will beat more rapidly and you'll smile for no reason.\nthey've decided its more fun if I don't.\nSo Tired D; Played Lazer Tag & Ran A "


    -------
    Last 300 letters of the data
    -------



    "ust had one a few weeks back....hopefully we will be back soon! wish you the best yo\nColombia is with an 'o'...“: We now ship to 4 countries in South America (fist pump). Please welcome Columbia to the Stunner Family”\n#GutsiestMovesYouCanMake Giving a cat a bath.\nCoffee after 5 was a TERRIBLE idea.\n"


    -------



```python
# test your code
x = """
I have a pen.\nI have an apple. \nAh\nApple pen.\n
"""
print(x)

split_to_sentences(x)
```

    
    I have a pen.
    I have an apple. 
    Ah
    Apple pen.
    
    





    ['I have a pen.', 'I have an apple.', 'Ah', 'Apple pen.']



Expected answer: 
```CPP
['I have a pen.', 'I have an apple.', 'Ah', 'Apple pen.']
```


```python
# Test your function
w3_unittest.test_split_to_sentences(split_to_sentences)
```

    [92m All tests passed


<a name='ex-2'></a>
### Exercise 2 - tokenize_sentences
The next step is to tokenize sentences (split a sentence into a list of words). 
- Convert all tokens into lower case so that words which are capitalized (for example, at the start of a sentence) in the original text are treated the same as the lowercase versions of the words.
- Append each tokenized list of words into a list of tokenized sentences.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>Use <a href="https://docs.python.org/3/library/stdtypes.html?highlight=split#str.lower" >str.lower</a> to convert strings to lowercase. </li>
    <li>Please use <a href="https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.punkt.PunktLanguageVars.word_tokenize" >nltk.word_tokenize</a> to split sentences into tokens.</li>
    <li>If you used str.split instead of nltk.word_tokenize, there are additional edge cases to handle, such as the punctuation (comma, period) that follows a word.</li>
</ul>
</p>



```python
def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)
    return tokenized_sentences
```


```python
# test your code
sentences = ["Sky is blue.", "Leaves are green.", "Roses are red."]
tokenize_sentences(sentences)
```




    [['sky', 'is', 'blue', '.'],
     ['leaves', 'are', 'green', '.'],
     ['roses', 'are', 'red', '.']]



### Expected output

```CPP
[['sky', 'is', 'blue', '.'],
 ['leaves', 'are', 'green', '.'],
 ['roses', 'are', 'red', '.']]
```


```python
# Test your function
w3_unittest.test_tokenize_sentences(tokenize_sentences)
```

    [92m All tests passed


<a name='ex-3'></a>
### Exercise 3 - get_tokenized_data


Use the two functions that you have just implemented to get the tokenized data.
- split the data into sentences
- tokenize those sentences


```python
def get_tokenized_data(data):
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences
```


```python
# test your function
x = "Sky is blue.\nLeaves are green\nRoses are red."
get_tokenized_data(x)
```




    [['sky', 'is', 'blue', '.'],
     ['leaves', 'are', 'green'],
     ['roses', 'are', 'red', '.']]



##### Expected outcome

```CPP
[['sky', 'is', 'blue', '.'],
 ['leaves', 'are', 'green'],
 ['roses', 'are', 'red', '.']]
```


```python
# Test your function
w3_unittest.test_get_tokenized_data(get_tokenized_data)
```

    [92m All tests passed


#### Split into train and test sets

Now run the cell below to split data into training and test sets.


```python
tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]
```


```python
print("{} data are split into {} train and {} test set".format(
    len(tokenized_data), len(train_data), len(test_data)))

print("First training sample:")
print(train_data[0])
      
print("First test sample")
print(test_data[0])
```

    47961 data are split into 38368 train and 9593 test set
    First training sample:
    ['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the', 'team', 'local', 'company', 'and', 'quality', 'production']
    First test sample
    ['that', 'picture', 'i', 'just', 'seen', 'whoa', 'dere', '!', '!', '>', '>', '>', '>', '>', '>', '>']


##### Expected output

```CPP
47961 data are split into 38368 train and 9593 test set
First training sample:
['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the', 'team', 'local', 'company', 'and', 'quality', 'production']
First test sample
['that', 'picture', 'i', 'just', 'seen', 'whoa', 'dere', '!', '!', '>', '>', '>', '>', '>', '>', '>']
```

<a name='ex-4'></a>
### Exercise 4 - count_words

You won't use all the tokens (words) appearing in the data for training.  Instead, you will use the more frequently used words.  
- You will focus on the words that appear at least N times in the data.
- First count how many times each word appears in the data.

You will need a double for-loop, one for sentences and the other for tokens within a sentence.


<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>If you decide to import and use defaultdict, remember to cast the dictionary back to a regular 'dict' before returning it. </li>
</ul>
</p>



```python
def count_words(tokenized_sentences):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    return word_counts
```


```python
# test your code
tokenized_sentences = [['sky', 'is', 'blue', '.'],
                       ['leaves', 'are', 'green', '.'],
                       ['roses', 'are', 'red', '.']]
count_words(tokenized_sentences)
```




    {'sky': 1,
     'is': 1,
     'blue': 1,
     '.': 3,
     'leaves': 1,
     'are': 2,
     'green': 1,
     'roses': 1,
     'red': 1}



##### Expected output

Note that the order may differ.

```CPP
{'sky': 1,
 'is': 1,
 'blue': 1,
 '.': 3,
 'leaves': 1,
 'are': 2,
 'green': 1,
 'roses': 1,
 'red': 1}
```


```python
# Test your function
w3_unittest.test_count_words(count_words)
```

    [92m All tests passed


#### Handling 'Out of Vocabulary' words

If your model is performing autocomplete, but encounters a word that it never saw during training, it won't have an input word to help it determine the next word to suggest. The model will not be able to predict the next word because there are no counts for the current word. 
- This 'new' word is called an 'unknown word', or <b>out of vocabulary (OOV)</b> words.
- The percentage of unknown words in the test set is called the <b> OOV </b> rate. 

To handle unknown words during prediction, use a special token to represent all unknown words 'unk'. 
- Modify the training data so that it has some 'unknown' words to train on.
- Words to convert into "unknown" words are those that do not occur very frequently in the training set.
- Create a list of the most frequent words in the training set, called the <b> closed vocabulary </b>. 
- Convert all the other words that are not part of the closed vocabulary to the token 'unk'. 




<a name='ex-5'></a>
### Exercise 5 - get_words_with_nplus_frequency

You will now create a function that takes in a text document and a threshold `count_threshold`.
- Any word whose count is greater than or equal to the threshold `count_threshold` is kept in the closed vocabulary.
- Returns the word closed vocabulary list. 


```python
def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    word_counts = count_words(tokenized_sentences)
    closed_vocab = [word for word, cnt in word_counts.items() if cnt >= count_threshold]
    return closed_vocab

```


```python
# test your code
tokenized_sentences = [['sky', 'is', 'blue', '.'],
                       ['leaves', 'are', 'green', '.'],
                       ['roses', 'are', 'red', '.']]
tmp_closed_vocab = get_words_with_nplus_frequency(tokenized_sentences, count_threshold=2)
print(f"Closed vocabulary:")
print(tmp_closed_vocab)
```

    Closed vocabulary:
    ['.', 'are']


##### Expected output

```CPP
Closed vocabulary:
['.', 'are']
```


```python
# Test your function
w3_unittest.test_get_words_with_nplus_frequency(get_words_with_nplus_frequency)
```

    [92m All tests passed


<a name='ex-6'></a>
### Exercise 6 - replace_oov_words_by_unk

The words that appear `count_threshold` times or more are in the closed vocabulary. 
- All other words are regarded as `unknown`.
- Replace words not in the closed vocabulary with the token `<unk>`.


```python
def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    vocabulary = set(vocabulary)
    replaced_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences
```


```python
tokenized_sentences = [["dogs", "run"], ["cats", "sleep"]]
vocabulary = ["dogs", "sleep"]
tmp_replaced_tokenized_sentences = replace_oov_words_by_unk(tokenized_sentences, vocabulary)
print(f"Original sentence:")
print(tokenized_sentences)
print(f"tokenized_sentences with less frequent words converted to '<unk>':")
print(tmp_replaced_tokenized_sentences)
```

    Original sentence:
    [['dogs', 'run'], ['cats', 'sleep']]
    tokenized_sentences with less frequent words converted to '<unk>':
    [['dogs', '<unk>'], ['<unk>', 'sleep']]


### Expected answer

```CPP
Original sentence:
[['dogs', 'run'], ['cats', 'sleep']]
tokenized_sentences with less frequent words converted to '<unk>':
[['dogs', '<unk>'], ['<unk>', 'sleep']]
```


```python
# Test your function
w3_unittest.test_replace_oov_words_by_unk(replace_oov_words_by_unk)
```

    [92m All tests passed


<a name='ex-7'></a>
### Exercise 7 - preprocess_data

Now we are ready to process our data by combining the functions that you just implemented.

1. Find tokens that appear at least count_threshold times in the training data.
1. Replace tokens that appear less than count_threshold times by "<unk\>" both for training and test data.


```python
def preprocess_data(train_data, test_data, count_threshold, unknown_token="<unk>"):
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token)
    return train_data_replaced, test_data_replaced, vocabulary
```


```python
# test your code
tmp_train = [['sky', 'is', 'blue', '.'],
     ['leaves', 'are', 'green']]
tmp_test = [['roses', 'are', 'red', '.']]

tmp_train_repl, tmp_test_repl, tmp_vocab = preprocess_data(tmp_train, 
                                                           tmp_test, 
                                                           count_threshold = 1
                                                          )

print("tmp_train_repl")
print(tmp_train_repl)
print()
print("tmp_test_repl")
print(tmp_test_repl)
print()
print("tmp_vocab")
print(tmp_vocab)
```

    tmp_train_repl
    [['sky', 'is', 'blue', '.'], ['leaves', 'are', 'green']]
    
    tmp_test_repl
    [['<unk>', 'are', '<unk>', '.']]
    
    tmp_vocab
    ['sky', 'is', 'blue', '.', 'leaves', 'are', 'green']


##### Expected outcome

```CPP
tmp_train_repl
[['sky', 'is', 'blue', '.'], ['leaves', 'are', 'green']]

tmp_test_repl
[['<unk>', 'are', '<unk>', '.']]

tmp_vocab
['sky', 'is', 'blue', '.', 'leaves', 'are', 'green']
```


```python
# Test your function
w3_unittest.test_preprocess_data(preprocess_data)
```

    [92m All tests passed


#### Preprocess the train and test data
Run the cell below to complete the preprocessing both for training and test sets.


```python
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, 
                                                                        test_data, 
                                                                        minimum_freq)
```


```python
print("First preprocessed training sample:")
print(train_data_processed[0])
print()
print("First preprocessed test sample:")
print(test_data_processed[0])
print()
print("First 10 vocabulary:")
print(vocabulary[0:10])
print()
print("Size of vocabulary:", len(vocabulary))
```

    First preprocessed training sample:
    ['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the', 'team', 'local', 'company', 'and', 'quality', 'production']
    
    First preprocessed test sample:
    ['that', 'picture', 'i', 'just', 'seen', 'whoa', 'dere', '!', '!', '>', '>', '>', '>', '>', '>', '>']
    
    First 10 vocabulary:
    ['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the']
    
    Size of vocabulary: 14821


##### Expected output

```CPP
First preprocessed training sample:
['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the', 'team', 'local', 'company', 'and', 'quality', 'production']

First preprocessed test sample:
['that', 'picture', 'i', 'just', 'seen', 'whoa', 'dere', '!', '!', '>', '>', '>', '>', '>', '>', '>']

First 10 vocabulary:
['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the']

Size of vocabulary: 14821
```

You are done with the preprocessing section of the assignment.
Objects `train_data_processed`, `test_data_processed`, and `vocabulary` will be used in the rest of the exercises.

<a name='2'></a>
## 2 - Develop n-gram based Language Models

In this section, you will develop the n-grams language model.
- Assume the probability of the next word depends only on the previous n-gram.
- The previous n-gram is the series of the previous 'n' words.

The conditional probability for the word at position 't' in the sentence, given that the words preceding it are $w_{t-n}\cdots w_{t-2}, w_{t-1}$ is:

$$ P(w_t | w_{t-n}\dots w_{t-1} ) \tag{1}$$

You can estimate this probability  by counting the occurrences of these series of words in the training data.
- The probability can be estimated as a ratio, where
- The numerator is the number of times word 't' appears after words t-n through t-1 appear in the training data.
- The denominator is the number of times word t-n through t-1 appears in the training data.


$$ \hat{P}(w_t | w_{t-n} \dots w_{t-1}) = \frac{C(w_{t-n}\dots w_{t-1}, w_t)}{C(w_{t-n}\dots w_{t-1})} \tag{2} $$


- The function $C(\cdots)$ denotes the number of occurence of the given sequence. 
- $\hat{P}$ means the estimation of $P$. 
- Notice that denominator of the equation (2) is the number of occurence of the previous $n$ words, and the numerator is the same sequence followed by the word $w_t$.

Later, you will modify the equation (2) by adding k-smoothing, which avoids errors when any counts are zero.

The equation (2) tells us that to estimate probabilities based on n-grams, you need the counts of n-grams (for denominator) and (n+1)-grams (for numerator).

<a name='ex-8'></a>
### Exercise 8 - count_n_grams
Next, you will implement a function that computes the counts of n-grams for an arbitrary number $n$.

When computing the counts for n-grams, prepare the sentence beforehand by prepending $n-1$ starting markers "<s\>" to indicate the beginning of the sentence.  
- For example, in the tri-gram model (n=3), a sequence with two start tokens "<s\>" should predict the first word of a sentence.
- So, if the sentence is "I like food", modify it to be "<s\> <s\> I like food".
- Also prepare the sentence for counting by appending an end token "<e\>" so that the model can predict when to finish a sentence.

Technical note: In this implementation, you will store the counts as a dictionary.
- The key of each key-value pair in the dictionary is a **tuple** of n words (and not a list)
- The value in the key-value pair is the number of occurrences.  
- The reason for using a tuple as a key instead of a list is because a list in Python is a mutable object (it can be changed after it is first created).  A tuple is "immutable", so it cannot be altered after it is first created.  This makes a tuple suitable as a data type for the key in a dictionary.
- Although for a n-gram you need to use n-1 starting markers for a sentence, you will want to prepend n starting markers in order to use them to compute the initial probability for the (n+1)-gram later in the assignment.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li> To prepend or append, you can create lists and concatenate them using the + operator </li>
    <li> To create a list of a repeated value, you can follow this syntax: <code>['a'] * 3</code> to get <code>['a','a','a']</code> </li>
    <li>To set the range for index 'i', think of this example: An n-gram where n=2 (bigram), and the sentence is length N=5 (including one start token and one end token).  So the index positions are <code>[0,1,2,3,4]</code>.  The largest index 'i' where a bigram can start is at position i=3, because the word tokens at position 3 and 4 will form the bigram. </li>
    <li>Remember that the <code>range()</code> function excludes the value that is used for the maximum of the range.  <code> range(3) </code> produces (0,1,2) but excludes 3. </li>
</ul>
</p>



```python
def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    n_grams = {}
    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence)
        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i:i+n]
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    return n_grams
```


```python
# test your code
# CODE REVIEW COMMENT: Outcome does not match expected outcome
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
print("Uni-gram:")
print(count_n_grams(sentences, 1))
print("Bi-gram:")
print(count_n_grams(sentences, 2))
```

    Uni-gram:
    {('<s>',): 2, ('i',): 1, ('like',): 2, ('a',): 2, ('cat',): 2, ('<e>',): 2, ('this',): 1, ('dog',): 1, ('is',): 1}
    Bi-gram:
    {('<s>', '<s>'): 2, ('<s>', 'i'): 1, ('i', 'like'): 1, ('like', 'a'): 2, ('a', 'cat'): 2, ('cat', '<e>'): 2, ('<s>', 'this'): 1, ('this', 'dog'): 1, ('dog', 'is'): 1, ('is', 'like'): 1}


Expected outcome:

```CPP
Uni-gram:
{('<s>',): 2, ('i',): 1, ('like',): 2, ('a',): 2, ('cat',): 2, ('<e>',): 2, ('this',): 1, ('dog',): 1, ('is',): 1}
Bi-gram:
{('<s>', '<s>'): 2, ('<s>', 'i'): 1, ('i', 'like'): 1, ('like', 'a'): 2, ('a', 'cat'): 2, ('cat', '<e>'): 2, ('<s>', 'this'): 1, ('this', 'dog'): 1, ('dog', 'is'): 1, ('is', 'like'): 1}
```

Take a look to the `('<s>', '<s>')` element in the bi-gram dictionary. Although for a bi-gram you will only require one starting mark, as in the element `('<s>', 'i')`, this `('<s>', '<s>')` element will be helpful when computing the probabilities using tri-grams (the corresponding count will be used as denominator).


```python
# Test your function
w3_unittest.test_count_n_grams(count_n_grams)
```

    [92m All tests passed


<a name='ex-9'></a>
### Exercise 9 - estimate_probability

Next, estimate the probability of a word given the prior 'n' words using the n-gram counts.

$$ \hat{P}(w_t | w_{t-n} \dots w_{t-1}) = \frac{C(w_{t-n}\dots w_{t-1}, w_t)}{C(w_{t-n}\dots w_{t-1})} \tag{2} $$

This formula doesn't work when a count of an n-gram is zero..
- Suppose we encounter an n-gram that did not occur in the training data.  
- Then, the equation (2) cannot be evaluated (it becomes zero divided by zero).

A way to handle zero counts is to add k-smoothing.  
- K-smoothing adds a positive constant $k$ to each numerator and $k \times |V|$ in the denominator, where $|V|$ is the number of words in the vocabulary.

$$ \hat{P}(w_t | w_{t-n} \dots w_{t-1}) = \frac{C(w_{t-n}\dots w_{t-1}, w_t) + k}{C(w_{t-n}\dots w_{t-1}) + k|V|} \tag{3} $$


For n-grams that have a zero count, the equation (3) becomes $\frac{1}{|V|}$.
- This means that any n-gram with zero count has the same probability of $\frac{1}{|V|}$.

Define a function that computes the probability estimate (3) from n-gram counts and a constant $k$.

- The function takes in a dictionary 'n_gram_counts', where the key is the n-gram and the value is the count of that n-gram.
- The function also takes another dictionary n_plus1_gram_counts, which you'll use to find the count for the previous n-gram plus the current word.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>To define a tuple containing a single value, add a comma after that value.  For example: <code>('apple',)</code> is a tuple containing a single string 'apple' </li>
    <li>To concatenate two tuples, use the '+' operator</li>
    <li><a href="" > words </a> </li>
</ul>
</p>



```python
def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + k * vocabulary_size
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    numerator = n_plus1_gram_count + k
    probability = numerator / denominator
    return probability
```


```python
# test your code
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
tmp_prob = estimate_probability("cat", ["a"], unigram_counts, bigram_counts, len(unique_words), k=1)

print(f"The estimated probability of word 'cat' given the previous n-gram 'a' is: {tmp_prob:.4f}")
```

    The estimated probability of word 'cat' given the previous n-gram 'a' is: 0.3333


##### Expected output

```CPP
The estimated probability of word 'cat' given the previous n-gram 'a' is: 0.3333
```


```python
# Test your function
w3_unittest.test_estimate_probability(estimate_probability)
```

    [92m All tests passed


#### Estimate probabilities for all words

The function defined below loops over all words in vocabulary to calculate probabilities for all possible words.
- This function is provided for you.


```python
def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>",  k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing
    
    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
    
    Returns:
        A dictionary mapping from next words to the probability.
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)    
    
    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + [end_token, unknown_token]    
    vocabulary_size = len(vocabulary)    
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, 
                                           n_gram_counts, n_plus1_gram_counts, 
                                           vocabulary_size, k=k)
                
        probabilities[word] = probability

    return probabilities
```


```python
# test your code
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)

estimate_probabilities(["a"], unigram_counts, bigram_counts, unique_words, k=1)
```




    {'i': 0.09090909090909091,
     'dog': 0.09090909090909091,
     'is': 0.09090909090909091,
     'a': 0.09090909090909091,
     'like': 0.09090909090909091,
     'this': 0.09090909090909091,
     'cat': 0.2727272727272727,
     '<e>': 0.09090909090909091,
     '<unk>': 0.09090909090909091}



##### Expected output

```CPP
{'cat': 0.2727272727272727,
 'i': 0.09090909090909091,
 'this': 0.09090909090909091,
 'a': 0.09090909090909091,
 'is': 0.09090909090909091,
 'like': 0.09090909090909091,
 'dog': 0.09090909090909091,
 '<e>': 0.09090909090909091,
 '<unk>': 0.09090909090909091}
```


```python
# Additional test
trigram_counts = count_n_grams(sentences, 3)
estimate_probabilities(["<s>", "<s>"], bigram_counts, trigram_counts, unique_words, k=1)
```




    {'i': 0.18181818181818182,
     'dog': 0.09090909090909091,
     'is': 0.09090909090909091,
     'a': 0.09090909090909091,
     'like': 0.09090909090909091,
     'this': 0.18181818181818182,
     'cat': 0.09090909090909091,
     '<e>': 0.09090909090909091,
     '<unk>': 0.09090909090909091}



##### Expected output

```CPP
{'cat': 0.09090909090909091,
 'i': 0.18181818181818182,
 'this': 0.18181818181818182,
 'a': 0.09090909090909091,
 'is': 0.09090909090909091,
 'like': 0.09090909090909091,
 'dog': 0.09090909090909091,
 '<e>': 0.09090909090909091,
 '<unk>': 0.09090909090909091}
```

#### Count and probability matrices

As we have seen so far, the n-gram counts computed above are sufficient for computing the probabilities of the next word.  
- It can be more intuitive to present them as count or probability matrices.
- The functions defined in the next cells return count or probability matrices.
- This function is provided for you.


```python
def make_count_matrix(n_plus1_gram_counts, vocabulary):
    # add <e> <unk> to the vocabulary
    # <s> is omitted since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]
    
    # obtain unique n-grams
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]        
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    
    # mapping from n-gram to row
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}    
    # mapping from next word to column
    col_index = {word:j for j, word in enumerate(vocabulary)}    
    
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix
```


```python
sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
bigram_counts = count_n_grams(sentences, 2)

print('bigram counts')
display(make_count_matrix(bigram_counts, unique_words))
```

    bigram counts



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>i</th>
      <th>dog</th>
      <th>is</th>
      <th>a</th>
      <th>like</th>
      <th>this</th>
      <th>cat</th>
      <th>&lt;e&gt;</th>
      <th>&lt;unk&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(&lt;s&gt;,)</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(i,)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(dog,)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(a,)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(like,)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(cat,)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(is,)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(this,)</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


##### Expected output

```CPP
bigram counts
          cat    i   this   a  is   like  dog  <e>   <unk>
(<s>,)    0.0   1.0  1.0  0.0  0.0  0.0   0.0  0.0    0.0
(a,)      2.0   0.0  0.0  0.0  0.0  0.0   0.0  0.0    0.0
(this,)   0.0   0.0  0.0  0.0  0.0  0.0   1.0  0.0    0.0
(like,)   0.0   0.0  0.0  2.0  0.0  0.0   0.0  0.0    0.0
(dog,)    0.0   0.0  0.0  0.0  1.0  0.0   0.0  0.0    0.0
(cat,)    0.0   0.0  0.0  0.0  0.0  0.0   0.0  2.0    0.0
(is,)     0.0   0.0  0.0  0.0  0.0  1.0   0.0  0.0    0.0
(i,)      0.0   0.0  0.0  0.0  0.0  1.0   0.0  0.0    0.0
```


```python
# Show trigram counts
print('\ntrigram counts')
trigram_counts = count_n_grams(sentences, 3)
display(make_count_matrix(trigram_counts, unique_words))
```

    
    trigram counts



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>i</th>
      <th>dog</th>
      <th>is</th>
      <th>a</th>
      <th>like</th>
      <th>this</th>
      <th>cat</th>
      <th>&lt;e&gt;</th>
      <th>&lt;unk&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(&lt;s&gt;, &lt;s&gt;)</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(this, dog)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(i, like)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(&lt;s&gt;, this)</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(a, cat)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(is, like)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(dog, is)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(&lt;s&gt;, i)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(like, a)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


##### Expected output

```CPP
trigram counts
              cat    i   this   a  is   like  dog  <e>   <unk>
(dog, is)     0.0   0.0  0.0  0.0  0.0  1.0   0.0  0.0    0.0
(this, dog)   0.0   0.0  0.0  0.0  1.0  0.0   0.0  0.0    0.0
(a, cat)      0.0   0.0  0.0  0.0  0.0  0.0   0.0  2.0    0.0
(like, a)     2.0   0.0  0.0  0.0  0.0  0.0   0.0  0.0    0.0
(is, like)    0.0   0.0  0.0  1.0  0.0  0.0   0.0  0.0    0.0
(<s>, i)      0.0   0.0  0.0  0.0  0.0  1.0   0.0  0.0    0.0
(i, like)     0.0   0.0  0.0  1.0  0.0  0.0   0.0  0.0    0.0
(<s>, <s>)    0.0   1.0  1.0  0.0  0.0  0.0   0.0  0.0    0.0
(<s>, this)   0.0   0.0  0.0  0.0  0.0  0.0   1.0  0.0    0.0
```

The following function calculates the probabilities of each word given the previous n-gram, and stores this in matrix form.
- This function is provided for you.


```python
def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix
```


```python
sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
bigram_counts = count_n_grams(sentences, 2)
print("bigram probabilities")
display(make_probability_matrix(bigram_counts, unique_words, k=1))
```

    bigram probabilities



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>i</th>
      <th>dog</th>
      <th>is</th>
      <th>a</th>
      <th>like</th>
      <th>this</th>
      <th>cat</th>
      <th>&lt;e&gt;</th>
      <th>&lt;unk&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(&lt;s&gt;,)</th>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>(i,)</th>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(dog,)</th>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(a,)</th>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>(like,)</th>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>(cat,)</th>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>(is,)</th>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(this,)</th>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
  </tbody>
</table>
</div>



```python
print("trigram probabilities")
trigram_counts = count_n_grams(sentences, 3)
display(make_probability_matrix(trigram_counts, unique_words, k=1))
```

    trigram probabilities



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>i</th>
      <th>dog</th>
      <th>is</th>
      <th>a</th>
      <th>like</th>
      <th>this</th>
      <th>cat</th>
      <th>&lt;e&gt;</th>
      <th>&lt;unk&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(&lt;s&gt;, &lt;s&gt;)</th>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>(this, dog)</th>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(i, like)</th>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(&lt;s&gt;, this)</th>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(a, cat)</th>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>(is, like)</th>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(dog, is)</th>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(&lt;s&gt;, i)</th>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>(like, a)</th>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
</div>


Confirm that you obtain the same results as for the `estimate_probabilities` function that you implemented.

<a name='3'></a>
## 3 - Perplexity

In this section, you will generate the perplexity score to evaluate your model on the test set. 
- You will also use back-off when needed. 
- Perplexity is used as an evaluation metric of your language model. 
- To calculate the perplexity score of the test set on an n-gram model, use: 

$$ PP(W) =\sqrt[N]{ \prod_{t=n+1}^N \frac{1}{P(w_t | w_{t-n} \cdots w_{t-1})} } \tag{4}$$

- where $N$ is the length of the sentence.
- $n$ is the number of words in the n-gram (e.g. 2 for a bigram).
- In math, the numbering starts at one and not zero.

In code, array indexing starts at zero, so the code will use ranges for $t$ according to this formula:

$$ PP(W) =\sqrt[N]{ \prod_{t=n}^{N-1} \frac{1}{P(w_t | w_{t-n} \cdots w_{t-1})} } \tag{4.1}$$

The higher the probabilities are, the lower the perplexity will be. 
- The more the n-grams tell us about the sentence, the lower the perplexity score will be. 

<a name='ex-10'></a>
### Exercise 10 - calculate_perplexity
Compute the perplexity score given an N-gram count matrix and a sentence. 

**Note:** For the sake of simplicity, in the code below, `<s>` is included in perplexity score calculation.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>Remember that <code>range(2,4)</code> produces the integers [2, 3] (and excludes 4).</li>
</ul>
</p>



```python
def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>', end_token='<e>', k=1.0):
    n = len(list(n_gram_counts.keys())[0])
    sentence = [start_token] * n + sentence + [end_token]
    sentence = tuple(sentence)
    N = len(sentence)
    product_pi = 1.0
    for t in range(n, N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
        product_pi *= 1 / probability
    perplexity = product_pi ** (1 / N)
    return perplexity
```


```python
# test your code

sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)


perplexity_train = calculate_perplexity(sentences[0],
                                         unigram_counts, bigram_counts,
                                         len(unique_words), k=1.0)
print(f"Perplexity for first train sample: {perplexity_train:.4f}")

test_sentence = ['i', 'like', 'a', 'dog']
perplexity_test = calculate_perplexity(test_sentence,
                                       unigram_counts, bigram_counts,
                                       len(unique_words), k=1.0)
print(f"Perplexity for test sample: {perplexity_test:.4f}")
```

    Perplexity for first train sample: 2.8040
    Perplexity for test sample: 3.9654



```python
# Test your function
w3_unittest.test_calculate_perplexity(calculate_perplexity)
```

    [92m All tests passed


### Expected Output

```CPP
Perplexity for first train sample: 2.8040
Perplexity for test sample: 3.9654
```

<b> Note: </b> If your sentence is really long, there will be underflow when multiplying many fractions.
- To handle longer sentences, modify your implementation to take the sum of the log of the probabilities.

<a name='4'></a>
## 4 - Build an Auto-complete System

In this section, you will combine the language models developed so far to implement an auto-complete system. 


<a name='ex-11'></a>
### Exercise 11 - suggest_a_word
Compute probabilities for all possible next words and suggest the most likely one.
- This function also take an optional argument `start_with`, which specifies the first few letters of the next words.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li><code>estimate_probabilities</code> returns a dictionary where the key is a word and the value is the word's probability.</li>
    <li> Use <code>str1.startswith(str2)</code> to determine if a string starts with the letters of another string.  For example, <code>'learning'.startswith('lea')</code> returns True, whereas <code>'learning'.startswith('ear')</code> returns False. There are two additional parameters in <code>str.startswith()</code>, but you can use the default values for those parameters in this case.</li>
</ul>
</p>


```python
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>", k=1.0, start_with=None):
    n = len(list(n_gram_counts.keys())[0])
    previous_tokens = ['<s>'] * n + previous_tokens
    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)
    suggestion = None
    max_prob = 0
    for word, prob in probabilities.items():
        if start_with and not word.startswith(start_with):
            continue
        if prob > max_prob:
            suggestion = word
            max_prob = prob
    return suggestion, max_prob
```


```python
# test your code
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)

previous_tokens = ["i", "like"]
tmp_suggest1 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0)
print(f"The previous words are 'i like',\n\tand the suggested word is `{tmp_suggest1[0]}` with a probability of {tmp_suggest1[1]:.4f}")

print()
# test your code when setting the starts_with
tmp_starts_with = 'c'
tmp_suggest2 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0, start_with=tmp_starts_with)
print(f"The previous words are 'i like', the suggestion must start with `{tmp_starts_with}`\n\tand the suggested word is `{tmp_suggest2[0]}` with a probability of {tmp_suggest2[1]:.4f}")
```

    The previous words are 'i like',
    	and the suggested word is `a` with a probability of 0.2727
    
    The previous words are 'i like', the suggestion must start with `c`
    	and the suggested word is `cat` with a probability of 0.0909


### Expected output

```CPP
The previous words are 'i like',
	and the suggested word is `a` with a probability of 0.2727

The previous words are 'i like', the suggestion must start with `c`
	and the suggested word is `cat` with a probability of 0.0909

```


```python
# Test your function
w3_unittest.test_suggest_a_word(suggest_a_word)
```

    [92m All tests passed


#### Get multiple suggestions

The function defined below loops over various n-gram models to get multiple suggestions.


```python
def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions
```


```python
# test your code
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
trigram_counts = count_n_grams(sentences, 3)
quadgram_counts = count_n_grams(sentences, 4)
qintgram_counts = count_n_grams(sentences, 5)

n_gram_counts_list = [unigram_counts, bigram_counts, trigram_counts, quadgram_counts, qintgram_counts]
previous_tokens = ["i", "like"]
tmp_suggest3 = get_suggestions(previous_tokens, n_gram_counts_list, unique_words, k=1.0)

print(f"The previous words are 'i like', the suggestions are:")
display(tmp_suggest3)
```

    The previous words are 'i like', the suggestions are:



    [('a', 0.2727272727272727), ('a', 0.2), ('a', 0.2), ('a', 0.2)]


#### Suggest multiple words using n-grams of varying length

Congratulations!  You have developed all building blocks for implementing your own auto-complete systems.

Let's see this with n-grams of varying lengths (unigrams, bigrams, trigrams, 4-grams...6-grams).


```python
n_gram_counts_list = []
for n in range(1, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_model_counts)
```

    Computing n-gram counts with n = 1 ...
    Computing n-gram counts with n = 2 ...
    Computing n-gram counts with n = 3 ...
    Computing n-gram counts with n = 4 ...
    Computing n-gram counts with n = 5 ...



```python
previous_tokens = ["i", "am", "to"]
tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest4)
```

    The previous words are ['i', 'am', 'to'], the suggestions are:



    [('be', 0.027665685098338604),
     ('have', 0.00013487086115044844),
     ('have', 0.00013490725126475548),
     ('i', 6.746272684341901e-05)]



```python
previous_tokens = ["i", "want", "to", "go"]
tmp_suggest5 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest5)
```

    The previous words are ['i', 'want', 'to', 'go'], the suggestions are:



    [('to', 0.014051961029228078),
     ('to', 0.004697942168993581),
     ('to', 0.0009424436216762033),
     ('to', 0.0004044489383215369)]



```python
previous_tokens = ["hey", "how", "are"]
tmp_suggest6 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest6)
```

    The previous words are ['hey', 'how', 'are'], the suggestions are:



    [('you', 0.023426812585499317),
     ('you', 0.003559435862995299),
     ('you', 0.00013491635186184566),
     ('i', 6.746272684341901e-05)]



```python
previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest7 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest7)
```

    The previous words are ['hey', 'how', 'are', 'you'], the suggestions are:



    [("'re", 0.023973994311255586),
     ('?', 0.002888465830762161),
     ('?', 0.0016134453781512605),
     ('<e>', 0.00013491635186184566)]



```python
previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest8 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with="d")

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest8)
```

    The previous words are ['hey', 'how', 'are', 'you'], the suggestions are:



    [('do', 0.009020723283218204),
     ('doing', 0.0016411737674785006),
     ('doing', 0.00047058823529411766),
     ('dvd', 6.745817593092283e-05)]


# Congratulations!

You've completed this assignment by building an autocomplete model using an n-gram language model!  

Please continue onto the fourth and final week of this course!
