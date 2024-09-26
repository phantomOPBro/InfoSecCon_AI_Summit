# Import required libraries
from keras.layers import Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import MaxPooling1D
import pandas as pd
import pickle

# Load training data
df = pd.read_csv(training data)

# Define the alphabet and the input and output sizes
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
input_size = 100
output_size = 14

# Create a tokenizer that converts characters to indices
tokenizer = Tokenizer(num_words=alphabet_size + 1, char_level=True)
tokenizer.fit_on_texts(alphabet)

# Convert the text data into sequences of token indices
X = tokenizer.texts_to_sequences(df[features]) #in our example this was df['file_name']

# Pad or truncate the sequences to have the same length
X = pad_sequences(X, maxlen=input_size)

# Convert the label column to a binary vector
y = df[label].values #for our example this was df['is_sensitive']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to float32 type
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_val = X_val.astype('float32')
y_val = y_val.astype('float32')

# Define the model parameters
vocab_size = 100  # The size of the vocabulary
max_len = 100  # The maximum length of the input sequence
embed_dim = 10  # The dimension of the embedding
num_filters = 2  # The number of filters for the convolutional layer, number of patterns to detect per kernel window
kernel_size = 12  # Window size
pool_size = 2  # Pool size for MaxPooling1D
hidden_dim1 = 32  # The dimension of the hidden layer
hidden_dim2 = 10  # The dimension of the hidden layer

# Define the model architecture
model = Sequential()
model.add(Embedding(alphabet_size + 1, embed_dim, input_length=max_len))
model.add(Conv1D(num_filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Conv1D(num_filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(num_filters, kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dim1, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_dim2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Set checkpoint so that you can restore the model to a best state if needed
checkpoint = ModelCheckpoint('InfoSecWorld_Log_CNN_Train.h5', verbose=1, monitor='val_loss',
                             save_best_only=True,
                             mode='auto')

#set early stopping to stop training if the model is not improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary, keep an eye on the number of parameters. This is far less than other text classification models tend to be
model.summary()

# Train the model on the training set and validate on the validation set
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                    callbacks=[checkpoint, early_stopping])



'''

============================================
Evaluate the model on UNSEEN DATA
============================================

'''
#now history.history is a dictionary containing the loss and accuracy of the model at each epoch

# Load test data
df = pd.read_csv(test_data)

# Normalize the features same as before
input_size = 100
X_test_file = pad_sequences(tokenizer.texts_to_sequences(feature), maxlen=input_size)
y_test = df[label]

# Convert test data to float32 type
X_test_file = X_test_file.astype('float32')
y_test = y_test.astype('float32')

# Evaluate the model
arr = model.evaluate(X_test_file, y_test)


'''

============================================
Now we are going to save the model and some useful info to keep on hand
============================================

'''


# Print loss and accuracy
loss = arr[0]
accuracy = arr[1]

# Write loss and accuracy to a file
with open('InfoSecWorld_Log_CNN_Train_lossacc.txt', 'w') as file:
    file.write(f'Loss: {loss}\n')
    file.write(f'Accuracy: {accuracy}\n')

# Save the tokenizer
with open('InfoSecWorld_Log_CNN_Train_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('InfoSecWorld_Log_CNN_Train_history.csv')

# Save the model
model.save('InfoSecWorld_Log_CNN_Train_model.h5')


'''

============================================
I also recommend getting used to onnx format
============================================

'''

# Save the model in ONNX format (generally recommended as industry standard for model exchange)
import tensorflow as tf
loaded_model = tf.keras.models.load_model('InfoSecWorld_Log_CNN_Train_model.h5')
tf.saved_model.save(loaded_model, 'saved_model')
!python -m tf2onnx.convert --saved-model 'saved_model' --output 'InfoSecWorld_Log_CNN_Train_model.onnx'


'''

============================================
Let's load the .h5 model and make a prediction. Later i will show you how to load the onnx file and make predictions
============================================

'''


# Load the model and make a prediction (from the .h5 file first and then ill show you the onyxx file)
from keras.models import load_model
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('InfoSecWorld_Log_CNN_Train_model.h5')

# Load the tokenizer
with open('InfoSecWorld_Log_CNN_Train_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

input_size = 100

# The filename to predict
filename = 'InfoSecWorld_is_the_best_conference_ever.pdf'

# Preprocess the filename
filename = filename.replace(' ', '').lower()
filename = np.array(tokenizer.texts_to_sequences([filename]))

# Pad the filename
filename_padded = pad_sequences(filename, maxlen=input_size)

# Make the prediction
prediction = model.predict(filename_padded)
print(prediction)


'''

============================================
Now let's load the onnx file and make a prediction
============================================

'''


# Load the model and make a prediction (from the .onnx file)
import onnxruntime

# Load the ONNX model
sess = onnxruntime.InferenceSession("InfoSecWorld_Log_CNN_Train_model.onnx")

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle

with open('InfoSecWorld_Log_CNN_Train_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

filename = "InfoSecWorld_is_the_best_conference_ever.pdf"

# Tokenize the filename
tokenized = tokenizer.texts_to_sequences([filename])

# Pad the tokenized filename
padded = pad_sequences(tokenized, maxlen=100)

# Convert the padded filename to a numpy array
input_data = np.array(padded).astype(np.float32)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name

# Get the output name for the ONNX model
output_name = sess.get_outputs()[0].name

# Use the ONNX model to make a prediction
result = sess.run([output_name], {input_name: input_data})
# Print the prediction
print("Prediction:", result[0][0][0])
