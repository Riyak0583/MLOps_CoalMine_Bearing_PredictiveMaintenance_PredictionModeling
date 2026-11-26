import pandas as pd
def data_ingestion():


    '''
    Title : 'CoalMine_Bearing_PredictiveMaintenance_PredictionModeling'

    Independent_Columns : 'record_id', 'timestamp', 'mine_id', 'conveyor_id', 'idler_id',
                          'vibration_rms', 'vibration_peak', 'vibration_kurtosis',
                          'vibration_crest_factor', 'bpfo_amplitude', 'bpfi_amplitude',
                          'bsf_amplitude', 'ftf_amplitude', 'bearing_temperature',
                          'acoustic_level', 'rotational_speed', 'ambient_temperature', 'humidity',
                          'dust_concentration', 'moisture_level', 'operating_hours',
                          'conveyor_load', 'belt_speed', 'motor_current', 'lubrication_age_hours',
                          'last_maintenance_hours', 'previous_failure_count',
                          'seal_condition_score', 'bearing_age_months'

    Dependent_Columns : 'failure_within_7_days'

    Model Type : 'Classification'

    Problem Statement : 'To develop a Machine Learning model that predicts whether a
                         conveyor bearing will fail within the next 7 days using historical
                         maintenance data, vibration measurements, sensor readings and
                         operational parameters'.
    '''

  

    # Loading the data using pandas function
    df1 = pd.read_csv(r'C:\MLOps_CoalMine_Bearing_PredictiveMaintenance_PredictionModeling\data\raw\preventative_maintainance_dataset.csv')
    df2 = pd.read_csv(r'C:\MLOps_CoalMine_Bearing_PredictiveMaintenance_PredictionModeling\data\interim\conveyor_failure_rates.csv')
    df3 = pd.read_csv(r'C:\MLOps_CoalMine_Bearing_PredictiveMaintenance_PredictionModeling\data\interim\daily_failure_rate.csv')
    df4 = pd.read_csv(r'C:\MLOps_CoalMine_Bearing_PredictiveMaintenance_PredictionModeling\data\interim\feature_statistics.csv')
    df5 = pd.read_csv(r'C:\MLOps_CoalMine_Bearing_PredictiveMaintenance_PredictionModeling\data\interim\mine_failure_rates.csv')
    df6 = pd.read_csv(r'C:\MLOps_CoalMine_Bearing_PredictiveMaintenance_PredictionModeling\data\interim\permutation_importance.csv')
    df7 = pd.read_csv(r'C:\MLOps_CoalMine_Bearing_PredictiveMaintenance_PredictionModeling\data\interim\vibration_deciles.csv')
    df8 = pd.read_csv(r'C:\MLOps_CoalMine_Bearing_PredictiveMaintenance_PredictionModeling\data\interim\vibration_threshold_curve.csv')

    
    dataframes = {
        'maintenance': df1,'conveyor_failure': df2,'daily_failure': df3,
        'feature_statistics': df4,'mine_failure': df5,'permutation_importance': df6,
        'vibration_deciles': df7,'vibration_threshold': df8
    }

    return dataframes