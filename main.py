
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_build
from src.model_prediction import model_predict
from src.model_evaluation import model_evaluate


def main():

    # Step1: Data Ingestion

    data = data_ingestion()
    df = data['maintenance']
    print(df.columns)

    #Step2 : Data Preprocessing
    X_train_resample, X_test_transform, y_train_resample, y_test = data_preprocessing(df)

    #Step3 : Model_Building
    model = model_build(X_train_resample, X_test_transform, y_train_resample, y_test)

    #Step4 : Model_Prediction
    y_pred = model_predict(model,X_test_transform)

    #Step5 : Model_Evaluation
    score = model_evaluate(y_test,y_pred)
    print(f'Model Classification Report : {score}')
main()