# Project Based Experiments

### DEVELOPED BY : VINUSH CV
### REGISTER NO : 212222230176

## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
 
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
```
### Load the MNIST dataset
```py
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
### Preprocess the data
```py
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)
```

### Split the dataset into training, validation, and test sets
```py
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train_categorical, test_size=0.1, random_state=42)
```

### Design MLP architecture
```py
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Adjust input shape
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```
### Compile the model
```py
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
### Train the model
```py
history = model.fit(X_train_split, y_train_split, epochs=10, batch_size=128, validation_data=(X_val_split, y_val_split))
```
### Evaluate the model on test set
```py
test_loss, test_acc = model.evaluate(X_test, y_test_categorical)
print("Test Accuracy:", test_acc)
```
### Make predictions on test set
```py
y_pred = np.argmax(model.predict(X_test), axis=-1)
```
### Calculate confusion matrix and classification report
```py
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
```
### Plot training and validation accuracy
```py
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
```
### Plot training and validation loss
```py
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```
### Load your image
```py
img_path = '/content/seven.png'  # Replace 'path_to_your_image.png' with the path to your image file
img = Image.open(img_path)
```
### Preprocess the image
```py
img_resized = img.resize((28, 28))
img_grayscale = img_resized.convert('L')
img_array = np.array(img_grayscale) / 255.0
img_input = img_array.reshape(1, 28, 28)  # Reshape the image array to match the model's input shape
```
### Make predictions
```py
predictions = model.predict(img_input)
predicted_class = np.argmax(predictions)
```
### Display the image and predicted class\
```py
plt.imshow(img_array, cmap='gray')
plt.title(f'Predicted Class: {predicted_class}')
plt.show()
```
## Output:
### Training Loss, Validation Loss Vs Iteration Plot:
![image](https://github.com/user-attachments/assets/66abd7bc-30db-4e49-aad3-dc9b1b6e3416)

![image](https://github.com/user-attachments/assets/965410d0-c55d-4580-9d0a-2431c4457412)

### Classification Report:
![image](https://github.com/user-attachments/assets/7433a4ad-766c-4f24-a36c-a3d827d67a1d)

### Confusion Matrix:
![image](https://github.com/user-attachments/assets/79dc1b8b-4971-42c0-8b99-ad27485da1ed)

### New Sample Data Prediction:
![image](https://github.com/user-attachments/assets/6f85d80e-4456-4cfa-aa1f-d6b5e95cf4f9)

## RESULT :
Thus, a Multilayer Perceptron (MLP) to classify handwritten digits in python is build.
