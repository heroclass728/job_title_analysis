# JobClassification

## Overview

This project is to develop the model to classify the job titles using NLP technology.

## Structure

- src

    The source code to process the text, train and test model.

- train_data

    The csv file with total job titles and tags, and the train, test csv files separated from that csv file.

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

- Please make the "train_data" folder in this project directory, copy the original csv file with all the job titles 
and tags and rename it "Job_titles.csv". At this time, if there is a null category in csv file, please replace that null
category with "nulll".

## Execution

- Prediction

    If you want only prediction of job titles with the trained model, after setting the value of PREDICTION_ONLY variable 
    True and the value of TEST_RATIO as you want (the default value is 0.3) in settings.py, run the following command.
    ```
        python3 app.py
    ``` 

- Training and Prediction

    If you want to train the model again and do the prediction, after setting the PREDICTION_ONLY False, and run the 
    above command. Then you will be able to get the newly trained model in utils/model.
