# Disaster Response Pipeline Project

## Table of contents
1.[Introduction](https://github.com/aparna-git-swe/Disaster_Response_Project#introduction)

2.[File Description](https://github.com/aparna-git-swe/Disaster_Response_Project#file-description)

3.[Installation](https://github.com/aparna-git-swe/Disaster_Response_Project#Installation)

4.[Instructions](https://github.com/aparna-git-swe/Disaster_Response_Project#Instructions)

5.[Screenshots](https://github.com/aparna-git-swe/Disaster_Response_Project#Screenshots)


## Introduction
This project is part of the Udacity's Data Scientist Nanodegree Program in collaboration with Figure Eight.<br/>
In this project, the pre-labeled disaster messages will be used to build a disaster response model that can categorize messages received in real time during a disaster event, so that messages can be sent to the right disaster response agency.<br/>

This project includes a web application where disaster response worker can input messages received and get classification results

## File Description

### Folder: data

**disaster_messages.csv** - real messages sent during disaster events (provided by Figure Eight)<br/>
**disaster_categories.csv** - categories of the messages<br/>
**process_data.py** - ETL pipeline used to load, clean, extract feature and store data in SQLite database<br/>
**ETL Pipeline Preparation.ipynb** - Jupyter Notebook used to prepare ETL pipeline<br/>
**DisasterResponse.db** - cleaned data stored in SQlite database


### Folder: models

**train_classifier.py** - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for later use<br/>
**classifier.pkl** - pickle file contains trained model<br/>
**ML Pipeline Preparation.ipynb** - Jupyter Notebook used to prepare ML pipeline

### Folder: app

**run.py** - python script to launch web application.<br/>
**Folder: templates** - web dependency files (go.html & master.html) required to run the web application.

## Installation

There should be no extra libraries required to install apart from those coming together with Anaconda distribution. There should be no issue to run the codes using Python 3.5 and above.

## Instructions

1. Run the following commands in the project's root directory to set up your database and model
   - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command  to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/  

## Screen Shots

1.Main page shows the Overview of Training Dataset & Distribution of Message Categories

![1  main page](https://user-images.githubusercontent.com/67683259/106398068-22713c00-6411-11eb-8335-d708d1754ac2.jpg)

2.After Entering 'Classify Message', we can see the category(ies) of which the message is classified to , highlighted in green

![3  classify result](https://user-images.githubusercontent.com/67683259/106398137-67956e00-6411-11eb-9900-04c9d664c5e1.jpg)

3.Run process_data.py for ETL pipeline,output cleaned data is stored in database<br/>

<img width="470" alt="process_data py output" src="https://user-images.githubusercontent.com/67683259/106398239-00c48480-6412-11eb-9650-20d50772f080.PNG">

4.Run train_classifier.py for ML pipeline<br/>

<img width="408" alt="train_classifier py output" src="https://user-images.githubusercontent.com/67683259/106399304-1e94e800-6418-11eb-8433-2f7ee0b4d1c2.PNG">


5.Run run.py in app's directory to run web app

<img width="441" alt="app output" src="https://user-images.githubusercontent.com/67683259/106398278-39645e00-6412-11eb-98c0-f2016baca8f9.PNG">



