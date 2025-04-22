# Neural-4
Q1:NLP Preprocessing Pipeline
This Python script performs basic NLP preprocessing on a given sentence, outputting:

Original Tokens: List of individual words and punctuation.
Tokens Without Stopwords: List after removing common stopwords.
Stemmed Words: List with words reduced to their root form using stemming.
Preprocessing Steps
The function follows these steps:

Tokenization: Split the sentence into individual words and punctuation.
Stopword Removal: Remove common English stopwords using the nltk.corpus.stopwords list.
Stemming: Apply stemming using nltk.stem.PorterStemmer to reduce each word to its root form.
Input Example
"NLP techniques are used in virtual assistants like Alexa and Siri."

Output Example
Original Tokens: ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.'] Tokens Without Stopwords: ['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri', '.'] Stemmed Words: ['NLP', 'techniqu', 'use', 'virtual', 'assist', 'like', 'Alexa', 'Siri', '.']

Requirements
Python 3.x
nltk library
Install using: !pip install nltk (if not already installed)
Setup Instructions for Google Colab
Install nltk library: In your Google Colab notebook, run the following command:

!pip install nltk
Download necessary resources: Before using the nltk tools for stopword removal and stemming, download the necessary datasets:

import nltk nltk.download('punkt') nltk.download('stopwords')

Run the preprocessing function with your sentence. Run the code: Once you've installed nltk and downloaded the resources, run the following code to see the results:

import nltk from nltk.tokenize import word_tokenize from nltk.corpus import stopwords from nltk.stem import PorterStemmer

Step 1: Tokenize the sentence into words
def preprocess_sentence(sentence): # Tokenizing the sentence tokens = word_tokenize(sentence) print("Original Tokens:", tokens)

# Step 2: Remove stopwords
stop_words = set(stopwords.words("english"))
tokens_without_stopwords = [word for word in tokens if word.lower() not in stop_words]
print("Tokens Without Stopwords:", tokens_without_stopwords)

# Step 3: Apply stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens_without_stopwords]
print("Stemmed Words:", stemmed_words)
Conclusion
This simple NLP preprocessing pipeline demonstrates basic text preprocessing tasks such as tokenization, stopword removal, and stemming. You can adapt this pipeline to process other sentences and texts as needed for further NLP tasks, like text classification, sentiment analysis, and more.

Q2: Named Entity Recognition with SpaCy
Introduction
This Python script demonstrates how to extract named entities from a sentence using the spaCy library. The task involves identifying and extracting entities like persons, organizations, dates, etc., from the sentence. For each entity, the script will output:

#Entity Text: The actual entity as detected in the sentence (e.g., "Barack Obama").

#Entity Label: The label associated with the entity (e.g., PERSON, DATE).

#Character Positions: The start and end positions of the entity within the sentence.

Input
The input sentence is:

"Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

Output Example
For the above sentence, the function will extract the following named entities:

Entity: Barack Obama, Label: PERSON, Start: 0, End: 12 Entity: 44th, Label: ORDINAL, Start: 22, End: 26 Entity: President, Label: ORG, Start: 31, End: 40 Entity: United States, Label: GPE, Start: 44, End: 58 Entity: Nobel Peace Prize, Label: ORG, Start: 63, End: 82 Entity: 2009, Label: DATE, Start: 86, End: 90

Setup Instructions for Google Colab
Install the spaCy library:

!pip install spacy Install the en_core_web_sm model:

!python -m spacy download en_core_web_sm Run the following code to extract named entities:

import spacy

Load the spaCy model
nlp = spacy.load("en_core_web_sm")

Input sentence
sentence = "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

Process the sentence
doc = nlp(sentence)

Print named entities
for ent in doc.ents: print(f"Entity: {ent.text}, Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}")

Requirements
Python 3.x

spaCy library

Install using: !pip install spacy

en_core_web_sm spaCy model

Install using: !python -m spacy download en_core_web_sm

#Troubleshooting If you encounter errors related to the model, ensure that en_core_web_sm is correctly installed. Ensure the spaCy library is installed and up-to-date by running !pip install spacy.

Conclusion
This script demonstrates how to use spaCy for named entity recognition (NER). You can adapt this code to process different sentences and detect a wide variety of named entities such as people, organizations, locations, and dates.

Q3: Scaled Dot-Product Attention
Introduction
This Python script demonstrates the implementation of the Scaled Dot-Product Attention mechanism. Given matrices Q (Query), K (Key), and V (Value), the following steps are performed:

Compute the dot product of Q and the transpose of K (i.e., QKᵀ).

Scale the result by dividing it by √d (where d is the key dimension).

Apply softmax to get attention weights.

Multiply the attention weights by V to get the output.

#Problem Description Given three matrices Q, K, and V, the task is to:

Compute the attention weights by applying the scaled dot-product formula.

Obtain the final output by multiplying these weights with the V matrix.

Display both the attention weights and the final output.

Matrices:
Q (Query): This matrix represents the queries that will be used to compute the attention.

K (Key): The key matrix is used to compute similarity with the queries.

V (Value): The value matrix holds the information that will be weighted based on the attention.

INPUT:
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]]) K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]]) V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

Explanation of the matrices:
Q: Query matrix with 2 rows and 4 columns.

K: Key matrix with 2 rows and 4 columns.

V: Value matrix with 2 rows and 4 columns.

Steps to Implement
Compute Dot Product (Q * Kᵀ):

Multiply Q with the transpose of K.

Scale by √d:

Divide the dot product result by √d, where d is the dimension of the key (i.e., the number of columns in K).

Apply Softmax:

Apply the softmax function on the scaled results to get the attention weights.

Multiply with V:

Multiply the attention weights by V to obtain the final output.

Requirements
Python 3.x

NumPy: Install using: !pip install numpy

#Troubleshooting Issue with Softmax: If you experience unexpected results from the softmax function, ensure you are applying it correctly for each row in the matrix.

Shape Mismatch: Make sure the matrices Q, K, and V are compatible in dimensions, i.e., Q.shape[1] must be equal to K.shape[1].

Conclusion
This code demonstrates the core mechanism behind the Scaled Dot-Product Attention used in transformers and attention-based models. You can adapt this function to process different matrices Q, K, and V depending on the application.

Q4:Sentiment Analysis using HuggingFace Transformers
Introduction
This notebook demonstrates how to perform Sentiment Analysis using the HuggingFace Transformers library. We will use a pre-trained sentiment analysis model and analyze the following input sentence:

"Despite the high price, the performance of the new MacBook is outstanding."

The program will load the pre-trained model, process the input sentence, and display:

Sentiment label (e.g., POSITIVE or NEGATIVE) Confidence score (a value between 0 and 1 that indicates the confidence of the model in its prediction)

Objective
The goal is to: Load a pre-trained sentiment analysis pipeline. Analyze the sentiment of a sentence. Print the sentiment label and confidence score.

Expected Output
After running the code, the output should clearly display: Sentiment: POSITIVE Confidence Score: 0.9985

Libraries Required
This code uses the following libraries:

transformers: For loading and using pre-trained models from HuggingFace. torch: To support the model inference and processing. numpy: For handling numerical operations.

To install these dependencies in Google Colab, run the following commands: !pip install transformers !pip install torch !pip install numpy

Steps to Implement
Install the Required Libraries You will need to install the HuggingFace Transformers library and Torch (PyTorch) in your Google Colab environment. !pip install transformers !pip install torch

Import the Required Libraries from transformers import pipeline

Load Pre-trained Sentiment Analysis Pipeline We will load a pre-trained sentiment analysis model using the pipeline API from HuggingFace. This model is already fine-tuned for sentiment classification tasks.

Load the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis') 4. Analyze the Input Sentence Use the pipeline to analyze the sentiment of the input sentence:

Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

Perform sentiment analysis
result = sentiment_analyzer(sentence)[0] 5. Display the Results The result contains both the label (Sentiment: POSITIVE or NEGATIVE) and the confidence score. We will print these results.

Display the results
print(f"Sentiment: {result['label']}") print(f"Confidence Score: {result['score']}")

Full Code Example

Install necessary libraries
!pip install transformers !pip install torch

Import necessary library
from transformers import pipeline

Load the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

Perform sentiment analysis
result = sentiment_analyzer(sentence)[0]

Display the results
print(f"Sentiment: {result['label']}") print(f"Confidence Score: {result['score']}")

Output
Sentiment: POSITIVE Confidence Score: 0.9985

Conclusion
This notebook demonstrates how to use the HuggingFace Transformers library to perform sentiment analysis using a pre-trained model. You can adapt this approach to analyze the sentiment of any text by modifying the input sentence.

#Troubleshooting Error: "ModuleNotFoundError: No module named 'transformers'" This error occurs if the transformers library is not installed. Install it using the following:

!pip install transformers Slow model inference HuggingFace models may be slow to load or inference, depending on the model size. Using a GPU in Google Colab can significantly speed up the process.

#Final Notes This notebook is useful for sentiment analysis tasks, and can be applied to a variety of text data, such as customer reviews, social media posts, etc.
