import dill
import json
import pandas as pd
import os


def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = dill.load(file)
    return model


def make_predictions(folder, model):

    prediction_results = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r') as file:

            data = json.load(file)
            df = pd.DataFrame(data, index=[0])
            prediction = model.predict(df)
            prediction_results.append({'car_id': filename.split('.')[0], 'pred': prediction[0]})

    return prediction_results

def predictions_to_csv(predictions, output_folder):

    predictions_df = pd.DataFrame(predictions)
    output_file = os.path.join(output_folder, 'predictions.csv')
    predictions_df.to_csv(output_file, index=False)


def predict():
    model_folder = 'data/models/'
    model_file = os.listdir(model_folder)[0]
    model = load_model(os.path.join(model_folder, model_file))
    predictions = make_predictions('data/test', model)
    predictions_to_csv(predictions, 'data/predictions')



if __name__ == '__main__':

    predict()

