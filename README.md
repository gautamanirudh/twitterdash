# Realtime Twitter Sentiment Analysis Dashboard
<!-- Twitter Analytics Dashboard -->

### Description 

Our project **Real Time Twitter Sentiment Analysis**, revolves around the idea of using unsuopervised Machine Learning approaches to classify the twitter data(tweets) into sentiment categories of POSITVE, NEGATIVE or NEUTRAL.

### Characteristic functionalities

* Analysis of Tweets from Twitter Usernames and Keywords.
* Classification of Tweets based on their sentiments in real-time.
* Interactive Charts and Graphs visualizing the corresponding twitter engagement.
* Options to choose custom input attributes like range of dates, maximum number of tweets to be fetched, etc.
* Dashboard presenting a complete twitter-performance-chart for the respective Userrname or keyword.
* Analysis of user engagement on the Twitter, based on different languages used, number of retweets and distribution of tweets over weekdays.


### Tech Stack 

* [Twint](https://github.com/twintproject/twint "Twint") for fetching tweets from Twitter in realtime.
* **Training the Sentiment Model**:
    * [NLTK](https://github.com/twintproject/twint "NLTK") provides several modeuls for data-preprocessing and Natural Language Processing in Python.
    * Twitter Sentiment Dataset from Kaggle was also reffered for gathering data to train the sentiment-model.
    * [ScikitLearn](https://github.com/twintproject/twint "ScikitLearn") provides useful model libraries. K-Means Clustering,
        * SkLeanr's TfIdf Vectorizer was used for preparing the embedded matrix.
    * [Gensim](https://github.com/twintproject/twint "Gensim") provides fast utilites for training NLP models and vector embeddings. 
        * Word2Vec model from gensim was used for vector embeddings.
    * [Pickle](https://github.com/twintproject/twint "Pickle") was used for serialiozing trained objects and using them for prediction and production.

* **Dashboard for Twitter Analysis**:
    * [Flask](https://github.com/twintproject/twint "Flask") was used as backend for Dashboard.
    * [Dash](https://github.com/twintproject/twint "Dash"), an HTML, CSS wrapper was used for laying out the UI for the Dashboard.
    * [Plotly](https://github.com/twintproject/twint "Plotly") was used for all charts, plots anbd graphical visualizations on the dashboard.

### Screenshots of the Dashboard

![dash](https://user-images.githubusercontent.com/56076028/106376026-4869f280-63b7-11eb-87fb-e1e3a6a4b817.jpeg)

![username](https://user-images.githubusercontent.com/56076028/106364418-dc56a280-6354-11eb-8bba-ee15e7cf6e31.jpeg)

![username1](https://user-images.githubusercontent.com/56076028/106364442-04460600-6355-11eb-9ce7-36540006fda4.jpeg)

![username2](https://user-images.githubusercontent.com/56076028/106364448-1758d600-6355-11eb-83d2-835529be9c72.jpeg)

![keyword](https://user-images.githubusercontent.com/56076028/106364458-29d30f80-6355-11eb-8d67-1ab1cc0faaf1.jpeg)

![keyword1](https://user-images.githubusercontent.com/56076028/106364473-3ce5df80-6355-11eb-8815-93a342eab3aa.jpeg)
