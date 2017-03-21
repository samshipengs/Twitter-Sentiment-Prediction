## Twitter Sentiment Analysis
Analyze sentiment of tweets i.e. positive, negative and neutral by applying *convolution neural network* on vector representations of words using *Word2Vec*. US Airline data is used in the demonstration.

### Data
[Airline Twitter sentiment](https://www.crowdflower.com/data-for-everyone/)
>Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as “late flight” or “rude service”).

### Prerequisites

* [lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html) - Create conv-net
* [nltk](http://www.nltk.org/install.html) - Data pre-processing
* [sklearn](http://scikit-learn.org/stable/install.html) - Provide useful tools e.g. stratified cross-validation


### Getting Started

Begin by creating a directory e.g. *twitter_sentiment* for stroing training data, Word2Vec model and CNN model, and set the FILE_PATH to this directory.
* data: contains training data (airline data in this case) and test data.
* word2vec: word embedding model is saved here.
* model: cnn model is saved here.

### Run
train cnn using *model_airline*,
```
jupyter notebook model_airline.ipynb
```
it could take some time to finish, and when it is done a *cnn.npz* file would be created.

make predictions on twitter data,
```
jupyter notebook predictions.ipynb
```
### Example
```python
airline_data = Data('Airline-Sentiment-2-w-AA.csv', FILE_PATH)
airline_df = airline_data.csv_df(['airline_sentiment', 'text']) # load data
airline_data.pre_process(airline_df) # pre-process data
airline_df.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline_sentiment</th>
      <th>text</th>
      <th>tokenized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>neutral</td>
      <td>What  said</td>
      <td>[said]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>positive</td>
      <td>plus youve added commercials to the experienc...</td>
      <td>[plus, youve, added, commercials, experience, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>neutral</td>
      <td>I didnt today Must mean I need to take anothe...</td>
      <td>[didnt, today, must, mean, need, take, another...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>negative</td>
      <td>its really aggressive to blast obnoxious ente...</td>
      <td>[really, aggressive, blast, obnoxious, enterta...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>negative</td>
      <td>and its a really big bad thing about it</td>
      <td>[really, big, bad, thing]</td>
    </tr>
  </tbody>
</table>
</div>

model gets trained

```
    Epoch 18 of 20 took 6.702s
      training loss:		0.386099
      validation loss:		0.914334
      validation accuracy:		64.32 %
    Epoch 19 of 20 took 6.602s
      training loss:		0.358274
      validation loss:		0.956272
      validation accuracy:		61.40 %
    Epoch 20 of 20 took 6.606s
      training loss:		0.317283
      validation loss:		0.948621
      validation accuracy:		63.72 %
 ```
 then use it to predict on tweets that mentions AirCanada. 
 
 Group the tweets based on the sentiment classified by CNN model, and we can find the most frequent words from each group,
 
```python
AC = GenerateData(FILE_PATH, 'AirCanada.csv', max_len_train=19)
AC.prepare_data(wv_size=600)
AC.get_result(n_preview=10, n_top = 20, name='AC_result',verbose=False)
```
```
===Positive===
[('great', 8), ('thanks', 7), ('thank', 4), ('home', 3), ('today', 3),
('help', 2), ('go', 2), ('lounge', 2), ('jetxd', 2), ('lhr', 2), 
('made', 2), ('yyz', 2), ('bag', 2), ('pbi', 2), ('bad', 2), 
('working', 2), ('visit', 2), ('service', 2), ('flying', 2), ('time', 2)]
===Neutral===
[('flight', 15), ('support', 11), ('travel', 10), ('trip', 9), ('big', 9), 
('vancouver', 8), ('change', 7), ('fly', 7), ('seat', 7), ('via', 6)]
===Negative===
[('flight', 35), ('time', 12), ('hours', 11), ('found', 10), ('worst', 9), 
('didnt', 9), ('actually', 9), ('board', 9), ('check', 9), ('service', 8), 
('hrs', 8), ('yvr', 7), ('never', 7), ('already', 7), ('plane', 7), 
('take', 7), ('delayed', 7), ('customer', 6), ('validb', 6), ('let', 6)]
```

Take a look of the context of some of the most frequent word used in negative grouped tweets,
```python
AC.check(word='worst', sentiment=3, n_view=10)
```
```
"@AirCanada you are the worst. I paid 4 a flight with a guaranteed seat. but of course u over booked. Now I have a... https://t.co/mxxQfETfH6"
"@AirCanada you are the worst. I paid 4 a flight with a guaranteed seat. but of course u over booked. Now I have a delay of 6 hours #boycott"
"@AirCanada you are the worst. Each experience worse than the last.  Too bad we don't have a choice of air carriers."
"@AirCanada is absolute worst airline. Rude staff, complete incompetence, left bags off flight - then yelled at us for asking after them"
"Just had the worst service experience ever on @AirCanada, I'll never flight again with you!"
"RT @mcjackson001: @AirCanada Absolute worst airline I have ever flown!!! NEVER flying them again. Take a cue from @SouthwestAir in the cust…"
"@AirCanada Absolute worst airline I have ever flown!!! NEVER flying them again. Take a cue from @SouthwestAir in the customer service dept."
"RT @gattaca: Hey @AirCanada just got off AC973, onboard service was the worst I’ve ever experienced, and I travel 200k per year. Pilot was…"
"Good morning, @AirCanada. I finally terminated our two-hour call last night after being on hold for 45 minutes. Worst customer service ever."
```

This could help airline to improve on relevant services.

### Reference
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
