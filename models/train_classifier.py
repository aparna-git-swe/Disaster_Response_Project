import sys
import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])
import re
import pandas as pd
import numpy as np
import pickle
import time
import joblib
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier




def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('disaster_messages',engine)
    df=df.dropna()
    X = df['message']
    y = df.drop(columns=['id','message','original','genre'])
    category_names=y.columns
    y=y.values
    
    return X,y,category_names
    


def tokenize(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text =text.replace(url,'urlplaceholder')
    
    tokens = [ lemmatizer.lemmatize(word).strip() for word in word_tokenize(text.lower()) if word.isalnum() and word not in  stop_words]
    return tokens


def build_model():
    """ 
    Args:
        grid_search : Boolean Flag 
    Returns:
        Model pipeline 
        
    """
    #creating pipeline
    pipeline = Pipeline([
     ('vect_count', CountVectorizer(tokenizer=tokenize)),
     ('tfidf',TfidfTransformer()),
     ('clf',MultiOutputClassifier(RandomForestClassifier()))
     ])

    
    #define parameters
    parameters={
        "vect_count__ngram_range": ((1, 1),(1,2)),
    "vect_count__max_df": (0.5,0.75,1.0),
    "tfidf__use_idf": (True,False),
    "clf__estimator__n_estimators": [10,20]
             }
        
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv
    
    return cv   

def evaluate_model(model, X_test, Y_test, category_names):
    """ To evaluate the trained model by calculating the classification report and accuracy score for each and every labels
    Args:
        model : Trained model
        X_test : Testing data
        Y_test : Testing labels
        category_names : Category/feature names
    Returns:
        Nothing
    """
    
    print(f"Evaluating the model on test data ....")
    
    ## predicting the labels
    Y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test[:, i], Y_pred[:, i]))
        print("Accuracy of %25s: %.2f" %(category_names[i], accuracy_score(Y_test[:, i], Y_pred[:,i])))



def save_model(model, model_filepath):
    """ Save the trained model
    Args:
        model : Trained model
        model_filepath : Range of stress factor
    Returns:
        Nothing
    """
    
    print(f" Saving the model ... ")
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()