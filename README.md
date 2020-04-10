# JobClassification

## Overview

This project is to develop the model to classify the job titles using NLP technology.

## Structure

- src

    The source code to process the text, train and test model.

- utils

    * The model to classify the job titles
    * The source code to manage the folder and file in this project and create the train and test csv data from the 
    original csv file.
    
- app

    The main execution file.
    
- requirements

    All the dependencies for this project.
    
- settings

    Several settings including the path of folders and files in this project and the ration between train and test data.
    
## Installation

- Environment

    Ubuntu 18.04, Python 3.6
    
- Dependency Installation

    * Please go ahead this project directory and run the following commands
    
    ```
    pip3 install -r requirements.txt
    ```
  
    * Then please go ahead the directory where the Spacy framework is installed and run the following command.
    ```
    python3 -m spacy download en_core_web_sm
    ```
  
- Model Configuration

    Please download the trained model from https://drive.google.com/file/d/1QPJNtk_BOGL3we5Gs53drUWy_oTqg7VV/view?usp=sharing 
    and copy it into utils/model.

## Execution

- Prediction one title

    Please run the following command.
    
    ```
        python3 app.py --job title "Title String to predict"
    ```
  
- Prediction csv file

    If you want only prediction of new job titles with the trained model, please copy the csv file to predict in the 
    project directory and set the following variables in settings file.
    
    * Please set PREDICTION_ONLY True.
    * Please set PREDICTION_ONE_TITLE False.
    * Please set NEW_PREDICTION_PATH the absolute path of the new csv file to predict.   
    
    And please go ahead project directory and run the following command.
    ```
        python3 app.py
    ``` 
  
    Then you can look at the predicted csv file, whose name is "predict.csv" in the project directory

- Training model and estimating accuracy

    If you want to train the model again and estimate its accuracy, please copy the seed data file as a csv in the 
    project directory, rename it "job_titles.csv" and set the following variables in settings file.
    
    * You can change the value of TEST_RATIO if you want, now its default is 0.3
    
    Then please go ahead the project directory and run the following command.
    
    ```
        python3 app.py
    ```
    
    After training, the accuracy of the model is displayed in terminal, and the model is saved in 
    /utils/model/job_title_model.joblib.

** NOTE: All the files to use in this project including new file to predict and seed data set file to train are 
csv files, not excel file.
And in all the csv files, the column names must be "Title" and "Tag", anything else is not allowed in project. 
Then you have to replace "null" value (if exists in file) with "unknown" in csv files you import, especially in Tag column,  
because Python recognizes the "null" value as any mathematical value, not string.
