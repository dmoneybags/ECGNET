# ECGNet by Daniel DeMoney
## Based upon the studies from this paper: https://www.nature.com/articles/s41597-020-0386-x
## Using this database https://figshare.com/collections/ChapmanECG/4560497/2
![alt text](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-020-0386-x/MediaObjects/41597_2020_386_Fig1_HTML.png)
## Background
An ECG is a measure of voltage relative to time. The above figure shows what a normal ECG would look like. An irregular heart is classified as a heartbeat which is abnormally fast, abnormally slow, or inconsistent relative to time. A link to the classifications and frequencies of the irregularities measured by the study can be found at https://www.nature.com/articles/s41597-020-0386-x/tables/3.
## Procedure
Being that the data is a matrix with the X axis representing time, and the y axis being voltage, I knew that a convolutional net would likely be the best for feature extraction and analysis. My basic hypothesis was that a convolutional kernel could be used to run over the data, generate activations representative of relevant features within the data, then pass those activations to a dense/series of dense layers, which could analyze the features and output a classification. In doing research, I found a paper with a similar goal as me, https://www.nature.com/articles/s41467-020-15432-4. The architecture for the neural net in the aforementioned paper can be seen below:
![alt text](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-020-15432-4/MediaObjects/41467_2020_15432_Fig3_HTML.png)
While I didn't entirely model my net after theirs, I took the idea of using a batch normalization layer to generalize the data and max pooling layers for feature analysis. The nature of the layers of my net is also completely linear, while theirs is more akin to a highway net, with multiple pathways for data.
## Results
After some tuning my model was able to reach accuracy of 85% after 4 epochs of data. This is 7% below the net within the aforementioned article, however they trained on a database 200 times the size of mine. As the number of samples is integral to preventing overfitting and increasing accuracy, I am very confident that if I tweak my code slightly to train on said data, I could possibly surpass their accuracy. However my computer is also extremely underpowered to support said training, and would likely take weeks to undergo a full cycle.
## Going Forward
Going forward, I plan to post a checkpoint of the trained model for individuals to try out and compare against other models. I also plan to find larger databases with more classifactions to train the model on. If anyone wishes to help with the repo that would be the most important thing to do.
