import sys
import pandas as pd
 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """ To get the raw data from two csv files one contains messages and another one all the labels to categorize 
        the paticular message both have unique column called id
    Args:
        messages_filepath : CSV file path for messages 
        categories_filepath : CSV file path for all the labeled categories 
    Returns:
        merged dataframe of messages and categories dataframes
    """
    dis_msg_df =pd.read_csv(messages_filepath)
    dis_cat_df= pd.read_csv(categories_filepath)
    merge_df = dis_msg_df.merge(dis_cat_df)
    return merge_df


def clean_data(df):
    
    """ To clean the data , which includes transform the categories data into different columns 
        and also assign the particular column names, also drops the duplicate rows from the cleaned data
    Args:
        df : dataframe
    Returns:
        cleaned dataframe
    """
    df_cat = df.categories.str.split(';',expand=True)
    row= df_cat[:1]
    cat_columns= [(str.split('-'))[0] for str in row.values[0]]
    df_cat= df_cat.applymap(lambda x : int(x.split('-')[-1]) )#rows with only values
    df_cat.columns=cat_columns
    # convert category values to just numbers 0 or 1
    for column in df_cat:
        # set each value to be the last character of the string
        df_cat[column] = df_cat[column].astype(str).str[-1]
        # convert column from string to numeric
        df_cat[column] = df_cat[column].astype(int)
    df= df.drop(columns='categories',axis=1)
    df=pd.concat([df,df_cat],axis=1)
    df=df.drop_duplicates()
    return df


def save_data(df, database_filename):
    
    """ Save the cleaned data as a SQL table in sqlite database
    Args:
        df : Cleaned data
        database_filename : database filepath
    Returns:
        Nothing
    """
   
    engine= create_engine(f"sqlite:///"+ database_filename)
    df.to_sql("disaster_messages", engine, index=False, if_exists='replace')#saving sql table to sqlite database


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()