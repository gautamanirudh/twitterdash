# Realtime Twitter Sentiment Analysis Dashboard
<!-- Twitter Analytics Dashboard -->

### Description 

Our project **Real Time Twitter Sentiment Analysis**, revolves around the idea of using unsuopervised Machine Learning approaches to classify the twitter data(tweets) into sentiment categories of POSITIVE, NEGATIVE or NEUTRAL.

### Characteristic functionalities

* Analysis of Tweets from Twitter Usernames and Keywords.
* Classification of Tweets based on their sentiments in real-time.
* Interactive Charts and Graphs visualizing the corresponding twitter engagement.
* Options to choose custom input attributes like range of dates, maximum number of tweets to be fetched, etc.
* Dashboard presenting a complete twitter-performance-chart for the respective Username or keyword.
* Analysis of user engagement on the Twitter, based on different languages used, number of retweets and distribution of tweets over weekdays.


### Tech Stack 

* [Twint](https://github.com/twintproject/twint "Twint") package is used for fetching tweets from Twitter in realtime.
* **Training the Sentiment Model**:
    * [NLTK](https://github.com/nltk/nltk "NLTK") provides several modules for data-preprocessing and Natural Language Processing in Python.
        * Preprocessing utilities from NTLK like stopwords, porter stemmer were used during the Text preprocessing stage in preparing the training dataset to be fed into the model.
    * [Twitter Sentiment Dataset](https://www.kaggle.com/kazanova/sentiment140 "Twitter Sentiment Dataset") from Kaggle is used for gathering data to train the sentiment-model.
    * [ScikitLearn](https://github.com/scikit-learn/scikit-learn "ScikitLearn") provides useful model libraries.
        * SkLeanr's TfIdf Vectorizer was used for preparing the embedded matrix.
        * Followed by it, K-Means Clustering model is used to cluster the semantically similar words from the embedded matrix and derive the cluster centers of three different sentiments.
    * [Gensim](https://github.com/RaRe-Technologies/gensim "Gensim") provides fast utilites for training NLP models and vector embeddings. 
        * Word2Vec model from gensim was used for vector embeddings.
    * [Pickle](https://github.com/python/cpython/blob/master/Lib/pickle.py "Pickle") was used for serializing trained models and using them for prediction and production. The trained models were pickled and dumped in the directory for further use.

* **Dashboard for Twitter Analysis**:
    * [Flask](https://github.com/pallets/flask "Flask") is used as backend for Dashboard.
    * [Dash](https://github.com/plotly/dash "Dash"), an HTML, CSS wrapper is used for laying out the UI for the Dashboard. Dash was predominantly used for setting up the Frontend of the Dashboard.
    * [Plotly](https://github.com/plotly "Plotly") is used for all charts, plots and graphical visualizations on the dashboard.

* **Determining the accuracy of the Sentiment Analysis Model**:
    For determining the accuracy, a dataset was choosen and its polarity was determined using pretrained Supervised ML model Vader Sentiment Analyser and then the F1 score was calculated using both the labelled data and the predicted data.
    * The accuracy of the model stands at: ```75.2%```
### Screenshots of the Dashboard

**Using a Twitter-Username for Analysing data**

![dash](https://user-images.githubusercontent.com/56076028/106376026-4869f280-63b7-11eb-87fb-e1e3a6a4b817.jpeg)

![username](https://user-images.githubusercontent.com/56076028/106364418-dc56a280-6354-11eb-8bba-ee15e7cf6e31.jpeg)

![username1](https://user-images.githubusercontent.com/56076028/106364442-04460600-6355-11eb-9ce7-36540006fda4.jpeg)

![username2](https://user-images.githubusercontent.com/56076028/106364448-1758d600-6355-11eb-83d2-835529be9c72.jpeg)


**Using a Keyword for Analysing data**

![keyword](https://user-images.githubusercontent.com/56076028/106364458-29d30f80-6355-11eb-8d67-1ab1cc0faaf1.jpeg)

![keyword1](https://user-images.githubusercontent.com/56076028/106364473-3ce5df80-6355-11eb-8815-93a342eab3aa.jpeg)


### Thought behind the Project

The project has several use cases in the industry ranging from, Analysing the sentiment of Users on Twitter for a particular product or service, to managing and proctoring the twitter engagement for tweets related a particular topic. The dashboard can act as a perfect tool for analysing market performance and further deciding the future of the service or product offered.

### Setup Process

For setting up the project on a local machine

* Fork this repository.
* Clone the repository using simple zip download or use the command
    ```
        git clone https://github.com/gautamanirudh/twitterdash.git
    ```
* Move to the master branch by using command
    ```
        git checkout  master
    ```
* Create a virtual environment for the project
    ```
        pip install virtualenv
        virtualenv -p /usr/bin/python3 env_name
    ```
* Activate the Virtual environment
    ```
       source env_name/bin/activate
    ```
    Once the virtual environment is activated, the name of your virtual environment will appear on left side of terminal. This will let you know that the virtual environment is currently active. 

* Install all the dependencies
    ```
       pip install -r requirements.txt
    ```

* To start the Dashboard app, run the command
    ```
        python app.py
    ```


**Above Steps are sufficient for running the dashboard and analyzing realtime twitter data sentiment performance. But, for running the preprocessing and training model files, nltk data has to be downloaded to access the utilities. For that use the command:**

    ```
        nltk.download()
    ```

