# Handwriting-Detection-with-TTS
A handwriting Recognition System with text to speech.
We used the already existing MNIST dataset for the numbers 0-9 and then also a free kaggle alphabet dataset

https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format

To train our model.
we achieved a traning accuracy of 98.67%, while a validation accuracy of 80%.
we then fed the model into a flask file which used gTTS to convert the predicted natural characters to speech.
