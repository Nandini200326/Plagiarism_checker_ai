#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[3]:


def preprocess_text(text):
    if isinstance(text, str):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Lowercase
        text = text.lower()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join(word for word in text.split() if word not in stop_words)
        return text
    else:
        return ''


# In[4]:


data_path = 'train_snli_converted.csv'
df = pd.read_csv(data_path)


# In[5]:


print("Columns:", df.columns)
df.head()


# In[6]:


df.rename(columns={'Premise': 'source_text', 'Hypothesis': 'plagiarized_text'}, inplace=True)
df['source_text'] = df['source_text'].apply(preprocess_text)
df['plagiarized_text'] = df['plagiarized_text'].apply(preprocess_text)
df['combined_text'] = df['source_text'] + ' ' + df['plagiarized_text']


# In[7]:


X_text = df['combined_text']
y = df['Label'].values  


# In[8]:


max_vocab_size = 10000
max_seq_length = 100
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
sequences = tokenizer.texts_to_sequences(X_text)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, y, test_size=0.2, random_state=42)


# In[10]:


embedding_dim = 100
model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_seq_length),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[11]:


epochs = 7
batch_size = 64
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=2
)


# In[16]:


train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


# In[21]:


def predict_plagiarism(source, suspect):
    source_p = preprocess_text(source)
    suspect_p = preprocess_text(suspect)
    combined = source_p + " " + suspect_p
    seq = tokenizer.texts_to_sequences([combined])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding='post', truncating='post')
    pred_prob = model.predict(padded)[0][0]
    pred_label = 1 if pred_prob >= 0.5 else 0
    print(f"Source Text: {source}")
    print(f"Suspect Text: {suspect}")
    print(f"Plagiarism Probability: {pred_prob:.4f}")
    print("Prediction:", "Plagiarized" if pred_label else "Not Plagiarized")
    print()


# In[22]:


#DEMO
predict_plagiarism(
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a lazy dog."
)


# In[23]:


#DEMO 
predict_plagiarism(
    "Artificial Intelligence is revolutionizing technology.",
    "Machine learning is transforming the tech industry."
)


# In[ ]:




