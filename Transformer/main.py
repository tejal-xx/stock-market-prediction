# Standard packages
import pandas as pd

# Scripts
from models import transf_params, TransformerModel
from dataset import GetDataset
from train import Classifier, plot_predictions
import analysis


def run(stock: str, model_type: str, stationary=True):
    df = analysis.get_data(stock)
    df["Company stock name"] = stock.split('/')[-1].split('.')[0]
    dataset = GetDataset(df)
    dataset.get_dataset(scale=False, stationary=stationary)
    train_data, test_data, train_data_len = dataset.split(
        train_split_ratio=0.8, time_period=30)
    train_data, test_data = dataset.get_torchdata()
    x_train, y_train = train_data
    x_test, y_test = test_data


    params = transf_params
    model = TransformerModel(params)

    clf = Classifier(model)
    clf.train([x_train, y_train], params=params)
    y_scaler = dataset.y_scaler
    predictions = clf.predict([x_test, y_test], y_scaler, data_scaled=False)
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    predictions.index = df.index[-len(x_test):]
    predictions['Actual'] = y_test[:-1]
    predictions.rename(columns={0: 'Predictions'}, inplace=True)
    if stationary:
        predictions = analysis.inverse_stationary_data(old_df=df, new_df=predictions,
                                                       orig_feature='Actual', new_feature='Predictions',
                                                       diff=12, do_orig=False)
    plot_predictions(df, train_data_len,
                     predictions["Predictions"].values, model_type)


if __name__ == '__main__':
    run('./data/apple.csv', 'transformer', True)
