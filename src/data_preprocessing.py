
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


def data_preprocessing(df):

    '''
     Step1: Drop duplicated values
     Step2: Feature Engineering (Optional)
     Step3: Split the dataset into Independent columns {X} and target column {y}
     Step4: Split the dataset into {train} Seen and {test} Unseen data
     Step5: Build E&E Pipeline for numerical and categorical
     Step6: Build numerical pipeline with imputation and scaling
     Step7: Build categorical pipeline with imputation and encoding
     Step8: Concat numerical and categorical pipeline using ColumnTransformer
     Step9: Fit transformer on train and test
     Step10: Fit_resample the data so that target column balance
     Step11: Return the preprocess train and test : X_train,X_test,y_train,y_test
    '''

    # Step 1: Drop duplicate values

    df.drop_duplicates(inplace=True)


    # Step 2: Feature Engineering


    # Step 3: Split dataset into X and y

    X = df.drop(columns=['record_id','timestamp','mine_id',
                         'conveyor_id','idler_id','failure_within_7_days'])

    y = df['failure_within_7_days']


    # Step 4: Split into training and testing data

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3, random_state=1)



    # Step 5: Segregate numerical columns and categorical columns

    numerical_col = X_train.select_dtypes(exclude='object').columns
    categorical_col = X_train.select_dtypes(include='object').columns


    # Step 6: Numerical pipeline

    numerical_pipeline = Pipeline([
        ('Imputation', SimpleImputer(strategy='median')),
        ('Scaling', MinMaxScaler())
    ])


    # Step 7: Categorical pipeline

    categorical_pipeline = Pipeline([
        ('Imputation', SimpleImputer(strategy='most_frequent')),
        ('Encoding', OneHotEncoder(handle_unknown='ignore'))
    ])


    # Step 8: ColumnTransformer

    transformer = ColumnTransformer([
        ('numerical_pipeline', numerical_pipeline, numerical_col),
        ('categorical_pipeline', categorical_pipeline, categorical_col)
    ])


    # Step 9: Fit and transform training data

    X_train_transform = transformer.fit_transform(X_train)


    # Step 10: Transform testing data

    X_test_transform = transformer.transform(X_test)


    # Step 11: Use target column to balance technique

    smote = SMOTE(random_state=1)
    X_train_resample, y_train_resample = smote.fit_resample(X_train_transform, y_train)


    # Step 12: Return processed data

    return X_train_resample, X_test_transform, y_train_resample, y_test
