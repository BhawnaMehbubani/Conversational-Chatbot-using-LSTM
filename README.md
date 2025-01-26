# Conversational Chatbot using LSTM

## Table of Contents
1. [Introduction](#introduction)  
2. [Project Objective](#project-objective)  
3. [Why LSTM for Chatbots?](#why-lstm-for-chatbots)  
4. [Dataset and Preprocessing](#dataset-and-preprocessing)  
5. [Pipeline Architecture Workflow](#pipeline-architecture-workflow)  
6. [Detailed Steps and Algorithms Used](#detailed-steps-and-algorithms-used)  
7. [Data Insights](#data-insights)  
8. [How to Run the Project](#how-to-run-the-project)  
9. [Results](#results)  
10. [Future Work and Improvements](#future-work-and-improvements)  



## Introduction

This project demonstrates the creation of a conversational chatbot using **Sequence-to-Sequence (Seq2Seq) Long Short-Term Memory (LSTM)** models. A chatbot built on deep learning offers human-like responses, making it ideal for applications like customer support, personal assistants, and more.



## Project Objective

The goal of this project is to:
- Build a chatbot capable of generating meaningful and contextually relevant responses.
- Train it using **Seq2Seq LSTM** models for tasks like **machine translation, speech recognition, and conversational AI**.
- Provide insights into the dataset used for training, model workflow, and evaluation.



## Why LSTM for Chatbots?

LSTM networks are a type of **Recurrent Neural Network (RNN)** that excels at processing and predicting sequence data by remembering information over long periods. For chatbots:
- **Memory cells** allow LSTMs to retain context while generating responses.
- They address the **vanishing gradient problem** of traditional RNNs.
- LSTMs can handle the sequential nature of conversational data effectively.



## Dataset and Preprocessing

**Dataset:**  
The chatbot uses the **Chatterbot Kaggle English Dataset**. It contains question-answer pairs from topics such as history, AI, and food.

### Key Preprocessing Steps:
1. **Parsing:** Extract questions and answers from `.yml` files.
2. **Cleaning:**
   - Remove punctuation and special characters.
   - Normalize case by converting text to lowercase.
   - Handle multi-line answers by concatenating sentences.
3. **Tokenizer Creation:** Map unique words to integer tokens.
4. **Data Arrays Preparation:**
   - **Encoder Input Data:** Tokenized questions padded to the maximum sequence length.
   - **Decoder Input Data:** Tokenized answers padded similarly.
   - **Decoder Output Data:** Shifted versions of `Decoder Input Data`.



## Pipeline Architecture Workflow

Here’s the detailed pipeline architecture, represented using **pipe symbols**:

```plaintext
Data Collection
    |
    v
+-----------------------------------+
| Kaggle Chatterbot Dataset         |
+-----------------------------------+
    |
    v
Data Preprocessing
    |
    +-----------------------------------------------------+
    |                                                     |
    v                                                     v
Question-Answer Parsing                             Text Cleaning
- Extract pairs from .yml files                   - Remove punctuation
- Concatenate multi-line answers                  - Lowercase conversion
                                                  - Tokenize and pad sequences
    |
    v
+---------------------------------------------------------------+
| Feature Engineering                                           |
| - Create Tokenizer for vocabulary mapping                    |
| - Build Encoder Input, Decoder Input, and Decoder Output      |
+---------------------------------------------------------------+
    |
    v
Seq2Seq Model Definition
    |
    +--------------------------------------------------------------+
    |                                                              |
    v                                                              v
Encoder Model                                               Decoder Model
- Embedding Layer                                           - Embedding Layer
- LSTM Layer                                                - LSTM Layer
- Dense Layer for state vectors                             - Dense Layer for predictions
    |
    v
Model Training
    |
    +--------------------------------------------------------------+
    |                                                              |
    v                                                              v
Loss Calculation                                           RMSProp Optimizer
- Categorical Crossentropy Loss                             - Update model weights
    |
    v
Model Evaluation
    |
    v
Inference Models
    |
    +--------------------------------------------------------------+
    |                                                              |
    v                                                              v
Encoder Inference Model                                   Decoder Inference Model
- Generates state vectors                                  - Generates response sequences
    |
    v
Chatbot Interaction
- Input questions
- Predict responses
- Generate conversational answers
```

```
Data Collection
    |
    v
+-----------------------------------+
| Kaggle Chatterbot English Dataset |
| - Downloaded .yml files with QA pairs  |
| - Topics: Food, History, AI, etc.|
+-----------------------------------+
    |
    v
Data Preprocessing
    |
    +-----------------------------------------------------+
    |                                                     |
    v                                                     v
Text Cleaning and Tokenization                          Handle Unwanted Data
- Remove unwanted characters (punctuation, special chars)   |
- Convert all text to lowercase                          |
- Remove stop words and short words (optional)            |
- Apply lemmatization or stemming (optional)             |
    |                                                     v
    v                                              Data Formatting
+----------------------------------------+         - Format data for padding
| Tokenization Process:                  |         - Map questions/answers to integer tokens
| - Tokenizer class from Keras           |         - Apply padding to ensure fixed input/output lengths
| - Create Vocabulary (words -> tokens)  |
+----------------------------------------+
    |
    v
Padding Sequences
    |
    v
+--------------------------------------------------+
| Pad input sequences to a uniform length (max_length)|
| - Encoder Inputs (questions)                     |
| - Decoder Inputs (answers)                       |
| - Decoder Outputs (shifted answers)              |
+--------------------------------------------------+
    |
    v
Seq2Seq Model Definition
    |
    +-----------------------------------------------------+
    |                                                     |
    v                                                     v
+----------------------------+             +---------------------------+
| Encoder Model              |             | Decoder Model             |
| - Input: encoder_input_data|             | - Input: decoder_input_data|
| - Embedding Layer          |             | - Embedding Layer         |
| - LSTM Layer (State vectors) |             | - LSTM Layer (with states)|
| - Dense Layer (Context)    |             | - Dense Layer (Output)    |
| - Output: Context Vectors  |             | - Output: Predicted Sequences|
+----------------------------+             +---------------------------+
    |
    v
Context Vectors passed to Decoder
    |
    v
+------------------------------------------------------------+
| Decoder uses encoder's LSTM states (h, c) to generate next |
| token in sequence. The process is iterative:               |
| - Prediction is made token-by-token                        |
| - Output is fed back as input to the next step             |
+------------------------------------------------------------+
    |
    v
Model Training
    |
    +--------------------------------------------------------------+
    |                                                              |
    v                                                              v
Loss Calculation                                              Optimizer (RMSProp)
- Calculate categorical_crossentropy loss                     - Update weights based on gradients
- Compute training accuracy                                     - Reduce loss iteratively
    |
    v
Epoch-wise Training
    |
    v
+---------------------------------------------------+
| Train for 150 epochs (or tune as per requirement) |
| - Model weights are updated using backpropagation  |
| - Training loss and accuracy are logged            |
+---------------------------------------------------+
    |
    v
Model Evaluation
    |
    v
+------------------------------+
| Evaluate model using validation|
| data to check accuracy        |
+------------------------------+
    |
    v
Inference Models
    |
    +-----------------------------------------------------------+
    |                                                           |
    v                                                           v
+----------------------------+             +----------------------------+
| Encoder Inference Model    |             | Decoder Inference Model    |
| - Input: Question (tokens) |             | - Input: Context vectors,   |
| - Output: State vectors (h,c)|             |   Decoder input token      |
+----------------------------+             | - Output: Predicted tokens |
    |                                         +----------------------------+
    v
Generate Response Iteratively
    |
    v
+----------------------------------------------------+
| - Convert Question to Tokens                     |
| - Predict Encoder Output (State vectors)         |
| - Use state vectors to feed Decoder model for the |
|   next token prediction                           |
| - Repeat until end of answer or max length is hit |
+----------------------------------------------------+
    |
    v
Display Final Response
    |
    v
+----------------------------+
| Return final chatbot response|
+----------------------------+
    |
    v
End of Interaction

```

## Detailed Steps and Algorithms Used

### 1. Data Extraction and Preprocessing:
- **Algorithm:** Python string operations, TensorFlow’s tokenizer API.
- **Goal:** Clean, tokenize, and pad the dataset for the Seq2Seq model.

### 2. Model Definition:
- **Encoder-Decoder Architecture:**  
  - Encoder compresses input sequences into a context vector (`h` and `c` state).
  - Decoder generates output sequences using context and previous outputs.
- **Algorithm:** LSTM layers with Keras Functional API.
- **Reason:** Seq2Seq models excel at sequence-based tasks.

### 3. Training:
- **Algorithm:** RMSprop optimizer + categorical cross-entropy loss.
- **Details:** 150 epochs with validation split for training accuracy (~96%).

### 4. Inference Models:
- **Encoder Inference Model:** Generates the context vector.
- **Decoder Inference Model:** Produces response sequences token by token.



## Data Insights

1. **Vocabulary Size:**  
   - Questions: ~12,000 unique tokens.  
   - Answers: ~15,000 unique tokens.

2. **Token Distribution:**  
   - Short questions/answers dominate (~5–10 tokens).  
   - Answers tend to include polite phrases like "thank you," "please," etc.

3. **Real-Life Observations:**
   - Repetitive questions often have slight variations in answers.  
   - Dataset bias is visible toward popular subjects like food and AI.  
   - Certain questions have multiple valid responses (e.g., greetings).



## How to Run the Project

### Prerequisites:
- Python 3.8 or above.
- Install required libraries:
  ```bash
  pip install numpy tensorflow keras pickle
  ```

### Steps to Run:
1. Clone the repository:
   ```bash
   git clone https://github.com/BhawnaMehbubani/Conversational-Chatbot-using-LSTM.git
   cd Conversational-Chatbot-using-LSTM
   ```
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook Zomato_Restaurant_Clustering_and_Sentiment_Analysis.ipynb
   ```
3. Interact with the chatbot using the **Talking with Chatbot** cell.


## Results

- **Training Accuracy:** 96% after 150 epochs.
- **Generated Responses:** Contextually relevant and coherent replies.
- **Examples:**
  - **Input:** "What is AI?"  
    **Output:** "AI stands for Artificial Intelligence."
  - **Input:** "How are you?"  
    **Output:** "I am fine, thank you."



## Future Work and Improvements

1. **Improve Response Variety:** Use Beam Search or Transformer-based models (e.g., BERT).  
2. **Add Context Awareness:** Include mechanisms to track conversation history.  
3. **Expand Dataset:** Add more domains and diversify the training data.  



