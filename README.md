A nueral net for abnormal heart beat classification using 12 lead ECG data in csv format. The Link to the data can be found at https://figshare.com/articles/dataset/ECGDataDenoised_zip/8378291?backTo=/collections/ChapmanECG/4560497

The net uses a traditional convolutional architecture of alternating conv2d layers and max pooling layers, with 2 dropout layers to prevent overfitting. With some tuning the accuracy of the net has been able to reach 85% after 4 epochs of training. In a near commit a checkpoint will be posted for users to load the pretrained model.

If training the model from scratch, download the whole folder on the github and place the data downloaded from the above link in the same directory and run the main.py file from the command line.

All code within the file uses python 3 and Tensorflow 2.6. There is currently no bash script to install the required libraries but one will be put in place in the next commit.
