import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

train = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")

# Gain some basic informations about the data
print("Dataset columns: ", train.columns)
print(train.info())

# Separate features from the target ("label" column here)
X = train.drop('label', axis = 1)
y = train.label                   #target attribute

#features values must range from 0 to 1
X = X / 255.0    
X_test = X_test/255.0

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test_final, y_val, y_test_final = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

# resize the data to it's original form to visualize
X_train_reshaped = X_train.to_numpy().reshape(-1, 28, 28)

plt.imshow(X_train_reshaped[0], cmap='gray')

#visualize the first 10 images and their label

plt.figure(figsize=(10, 15))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.imshow(X_train_reshaped[i], cmap='gray')
    plt.title(f"Label: {y_train.iloc[i]}")
plt.tight_layout()


# try three models with different architecture of layers
input_shape = [X_train.shape[1]]

# Model_1: Balanced architecture with moderate complexity
# This model has a few hidden layers with moderate units.

model_1 = Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    Dense(units = 350, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    Dense(units = 165, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    Dense(units = 64, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    Dense(units = 10, activation = 'softmax')
])

# Model_2: Simpler architecture with fewer parameters
# Suitable for faster training with lower capacity to learn complex patterns.
model_2 = Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    Dense(units = 80, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    Dense(units = 10, activation = 'softmax')
])

# Model_3: Deeper architecture with many more units and layers
# Designed to handle more complex datasets; may require more computation time.
model_3 = Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    Dense(units = 560, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    Dense(units = 700, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    Dense(units = 430, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    Dense(units = 120, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    Dense(units = 10, activation = 'softmax')
])

# Decreasing learning rate by 5% every epoch to improve convergence and prevent overshooting the minimum
lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95**epoch)

# evaluate each model 

models = [model_1, model_2, model_3]

# Early stopping to prevent overfitting by monitoring validation accuracy
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy for better stopping
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)


for model in models:
    # This loss function is suitable for multiclass classification problems with integer-encoded target labels.
    # Adam is an advanced optimizer combining momentum and adaptive learning rate, ensuring faster and more stable convergence.
    model.compile(loss = SparseCategoricalCrossentropy(from_logits= False),
               optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
               metrics = ['accuracy'])
    
    print(f"Training model_{models.index(model) + 1}...")
    history = model.fit(X_train, y_train, epochs = 50,
           validation_data = (X_val, y_val),
           batch_size = 64,
           callbacks=[early_stopping, lr_schedule],
           shuffle=True,
           verbose = 1
           )
    
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot(title=f"Model {models.index(model) + 1} Performance")

    #history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
    
    print("Done!\n")

# Evaluate each model's performance on validation data
for idx, model in enumerate(models):
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Model_{idx + 1} - Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# It's seems like model_1 performs better
best_model = model_1

# Evaluate model on validation data and generate predictions
y_pred = best_model.predict(X_val).argmax(axis=1)

# Add Confusion Matrix and Classification Report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap='Blues')
plt.title("Confusion Matrix for Best Model")
plt.show()

# Train on training set using explicit validation data
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping, lr_schedule], batch_size=64, epochs=50)

history_df = pd.DataFrame(history.history)

# Plot only if the key exists in the DataFrame to avoid KeyError
if 'val_loss' in history_df.columns:
    history_df.loc[:, ['loss', 'val_loss']].plot(title="Best Model - Cross-entropy")
else:
    history_df.loc[:, ['loss']].plot(title="Best Model - Cross-entropy")

if 'val_accuracy' in history_df.columns:
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Best Model - Accuracy")
else:
    history_df.loc[:, ['accuracy']].plot(title="Best Model - Accuracy")

# Evaluate best model on the final test set
final_eval_loss, final_eval_accuracy = best_model.evaluate(X_test_final, y_test_final)
print(f"Final Test Set Loss: {final_eval_loss}")
print(f"Final Test Set Accuracy: {final_eval_accuracy}")

# Generate predictions on the test set
y_test_pred = best_model.predict(X_test_final).argmax(axis=1)

# Add Confusion Matrix and Classification Report for the test set
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test_final, y_test_pred, cmap='Blues')
plt.title("Confusion Matrix for Test Set")
plt.show()

print(classification_report(y_test_final, y_test_pred))

test_predictions = best_model.predict(X_test)

ImageId = []
Label = []
for i in range(len(test_predictions)):
    ImageId.append(i+1)
    Label.append(test_predictions[i].argmax())
    
submissions=pd.DataFrame({"ImageId": ImageId,
                         "Label": Label})
submissions.to_csv("submission.csv", index=False, header=True)

import nbformat

def extract_code_from_ipynb(ipynb_file, output_file):
    with open(ipynb_file, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)
        
    code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, code in enumerate(code_cells, 1):
            file.write(code)
            file.write('\n\n')

# Replace 'notebook.ipynb' and 'output.py' with your file names
extract_code_from_ipynb('ann2.ipynb', 'output.py')



