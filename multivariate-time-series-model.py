import pandas as pd
import matplotlib.pyplot as plt
import traceback


class DataModel:
    def __init__(self, indat):
        self.indat = indat

    def df_to_cnn_rnn_format(
            self,
            train_size=0.5,
            look_back=5,
            target_column='target',
            scale_X=True):
        """
        TODO: output train and test datetime
        Input is a Pandas DataFrame.
        Output is a np array in the format of (samples, timesteps, features).
        Currently this function only accepts one target variable.
        Usage example:
        # variables
        df = data # should be a pandas dataframe
        test_size = 0.5 # percentage to use for training
        target_column = 'c' # target column name, all other columns are taken as features
        scale_X = False
        look_back = 5 # Amount of previous X values to look at when predicting the current y value
        """
        df = self.indat.copy()

        # Make sure the target column is the last column in the dataframe
        df['target'] = df[target_column]  # Make a copy of the target column
        # Drop the original target column
        df = df.drop(columns=[target_column])

        target_location = df.shape[1] - 1  # column index number of target
        # the index at which to split df into train and test
        split_index = int(df.shape[0] * train_size)

        # ...train
        X_train = df.values[:split_index, :target_location]
        y_train = df.values[:split_index, target_location]

        # ...test
        # original is split_index:-1
        X_test = df.values[split_index:, :target_location]
        # original is split_index:-1
        y_test = df.values[split_index:, target_location]

        # Scale the features
        if scale_X:
            scalerX = StandardScaler(
                with_mean=True, with_std=True).fit(X_train)
            X_train = scalerX.transform(X_train)
            X_test = scalerX.transform(X_test)

        # Reshape the arrays
        samples = len(X_train)  # in this case 217 samples in the training set
        num_features = target_location  # All columns before the target column are features

        samples_train = X_train.shape[0] - look_back
        X_train_reshaped = np.zeros((samples_train, look_back, num_features))
        y_train_reshaped = np.zeros((samples_train))

        for i in range(samples_train):
            y_position = i + look_back
            X_train_reshaped[i] = X_train[i:y_position]
            y_train_reshaped[i] = y_train[y_position]

        samples_test = X_test.shape[0] - look_back
        X_test_reshaped = np.zeros((samples_test, look_back, num_features))
        y_test_reshaped = np.zeros((samples_test))

        for i in range(samples_test):
            y_position = i + look_back
            X_test_reshaped[i] = X_test[i:y_position]
            y_test_reshaped[i] = y_test[y_position]

        return X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped


class viz:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def ts_plot(self):
        plt.figure(figsize=(12, 7))
        plt.plot(
            *self.args,
            color=self.kwargs["color"],
            label=self.kwargs['label'],
            alpha=self.kwargs['alpha'])
        plt.xlabel('Datetime [-]', fontsize=20)
        #plt.ylabel(
        #    r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' %
        #    (magnitude), fontsize=14)
        # plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14)
        plt.title('Mean gas usage of 52 houses', fontsize=14)
        plt.legend(
            loc='upper left',
            borderaxespad=0,
            frameon=False,
            fontsize=14,
            markerscale=3)
        plt.tight_layout()
        plt.savefig('data/figures/available data.png', dpi=1200)
        plt.show()


class ExamineDataModel:
    def __init__(self, in_file):
        self.in_file = in_file

    def __call__(self):
        df = pd.read_csv(self.in_file, delimiter="\t")
        df = df.set_index(['datetime'])
        data = df.copy()
        columns_to_category = ['hour', 'dayofweek', 'season']
        data[columns_to_category] = data[columns_to_category].astype(
            'category')  # change datetypes to category
        # One hot encoding the categories
        data = pd.get_dummies(data, columns=columns_to_category)
        data['gasPower_copy'] = data['gasPower']
        _ = viz(
            data.index,
            data['gasPower'],
            '.-',
            color='red',
            label='Original data',
            alpha=0.5).ts_plot()


def main():
    try:
        _ = ExamineDataModel('data/house_data_processed.csv')()
    except Exception as exp:
        print(traceback.format_exc())
        print(exp.__class__, " occurred.")


if __name__ == "__main__":
    main()
