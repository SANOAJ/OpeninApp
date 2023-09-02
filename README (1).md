
# Project Title
Title: English to Hindi Neural Machine Translation Using TensorFlow

Description: In this deep learning project, we build a neural machine translation(NMT) model that translates English text to Hinligh text using TensorFlow and Keras. The project includes data preprocessing, model architecture design, training, and saving the model for future translation tasks. The provided code template serves as a starting point for developers interested in building their own English to Hindi translation systems.





## Data Set

The dataset comprises a total of 2979 data points, each containing English phrases along with their corresponding Hindi translations. To begin the data analysis process, the dataset must first be extracted from the provided Zip archive.

The dataset comes from http://www.manythings.org/anki/, where you may nd tab delimited bilingual sentence pairs in
different les based on the source and target language of your choice.

## Preprocessing
The data preprocessing phase involved several key steps to determine the optimal sequence lengths for both English and Hindi phrases in our dataset. These steps were crucial to enhance the efficiency of our LSTM model training while reducing unnecessary complexity. Here's a formal description of the preprocessing steps:

1. Data Loading: 
The dataset was loaded and prepared for analysis.

2. Remove Punctuation:
We'll now get rid of all the punctuations in both the English and Hindi phrases by using maketrans.

3. Convert to Lower case: Converting the text to lower case

4. Histogram Analysis: A histogram was created, where one axis represented the size (length) of data points, and the other axis indicated the number of data points with that specific size.

5. Optimal Length Determination: Find out the maximum length of of English text and Hindi text.

From our data We can see that the maximum length sequence in english is 22 and  maximum length sequence in Hindi is 25.

5. Tokenize: Tokenization is the process of converting each word in the vocabulary into an integer based on frequency of occurence

6. Encoding: Encode and pad sequences,
Encoding means replacing each word with its corresponding number Padding essentially means adding zeros to make the length of every sequence equal.






## Screenshots
![App Screenshot](https://drive.google.com/file/d/1j6Hst823UEfzc2uPz5f6lZRpQnOs20vW/view?usp=sharing)


![App Screenshot](https://drive.google.com/file/d/1kTQOTEmBOf5aAJQHA9RqQOo8Q6Gw7VI-/view?usp=sharing)





## Split The Dataset
Train-Test Split:

The dataset is split into training and testing sets, with 80% used for training and 20% for testing.

## Model
1. Encoder
Encoder takes the English data as input and converts it into vectors that is passed to an LSTM model for training. We discard the encoder output and only keep the states.

2. Model Description
Now we'll build the Sequential model. The first layer is the embedding layer which projects each token in an N dimensional vector space LSTM is the artificial recurrent neural net architecture. It can not only proces past data but take feedback from future data as well.

In the second LSTM layer, we have set return sequences as True becuase we need outputs of all hidden units and not just the last one.

3. Model Architecture:
The core of the model is defined in the build_model function, which creates the neural network architecture. Here's a breakdown of the layers:

Embedding Layer: This layer converts input tokens (words) into dense vectors. It helps the model understand the semantics of words in the input sequence.

LSTM Layers: Long Short-Term Memory layers are used for sequence modeling. There are two LSTM layers in the encoder-decoder architecture.

RepeatVector Layer: This layer repeats the encoder's output sequence, making it compatible with the decoder input.

LSTM Layer (Decoder): Another LSTM layer in the decoder processes the repeated sequence from the encoder.

Dense Layer: The output layer with a softmax activation function maps the model's predictions to the vocabulary space for generating translated text.

3. Model Compilation
The model is compiled using the RMSprop optimizer and sparse categorical cross-entropy loss. RMSprop is an optimization algorithm, and sparse categorical cross-entropy is used for training models that predict class labels.
## Train Model
The model was trained exclusively on a standard PC, without utilizing GPU acceleration. Due to limited computational resources, the training process was conducted in multiple stages. Each training session spanned 30 epochs and took approximately 30 minutes to complete.Ultimately, the model achieved an accuracy of approximately 67%.

Upon completing the training for English to Hinglish translation on the dataset, the model attained an average BLEU score of 67.00262030400634%.

After completing the traing I saved the trained model as "Trained_model.pkl" format.
## Model Test
The trained model was loaded from a local file named "Trained_model.pkl" for evaluation. It is important to note that this model was developed without GPU utilization and trained in multiple segments due to limited computational resources.




## Potential & Further Improvements
The potential for further improvement in accuracy exists through the addition of more data points. I have currently utilized 2979 data points, and with the incorporation of additional data, I can enhance the model's precision.
## Conclusion
In conclusion, upon observing the performance of my trained model, it is evident that it possesses the capability to generate text in the Hinglish format. It is worth noting that the predictions produced by the model may exhibit minor variations in response to different circumstances or input data. However, it is essential to emphasize that these variations do not compromise the fundamental meaning and coherence of the generated text. The model effectively bridges the gap between English and Hinglish, demonstrating its utility for text translation tasks in this specific linguistic context.