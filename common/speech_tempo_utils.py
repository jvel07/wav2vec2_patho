import pandas as pd

speech_tempo_train = pd.read_csv('../data/bea-base-train-flat/speech_tempo_train_5000.csv')
speech_tempo_dev = pd.read_csv('../data/bea-base-train-flat/speech_tempo_dev_5000.csv')

no_pause_speech_train = speech_tempo_train[['name', 'no_pause_speech']]
no_pause_speech_dev = speech_tempo_dev[['name', 'no_pause_speech']]

# Define threshold values and corresponding labels
thresholds = [0.00035, 0.0006, 0.0009, 0.0012]
labels = ['slow', 'midslow', 'normal', 'fast']

# Define function to apply to each element of the column
def label_element(val):
    if val < thresholds[0]:
        return labels[0]
    elif val >= thresholds[-1]:
        return labels[-1]
    else:
        for i in range(len(thresholds)-1):
            if thresholds[i] <= val < thresholds[i+1]:
                return labels[i+1]


tempo_targets = ['whole_speech', 'no_pause_speech']

for target in tempo_targets:

    # drop column
    if target == 'whole_speech':
        to_drop = ['no_pause_speech', 'length', 'whole_speech']
    elif target == 'no_pause_speech':
        to_drop = ['whole_speech', 'length', 'no_pause_speech']

    # Use apply method to apply the label_element function to each element of the column
    speech_tempo_train['speed'] = speech_tempo_train[target].apply(label_element)
    speech_tempo_train_new = speech_tempo_train.drop(to_drop, axis=1)

    speech_tempo_dev['speed'] = speech_tempo_dev[target].apply(label_element)
    speech_tempo_dev_new = speech_tempo_dev.drop(to_drop, axis=1)

    speech_tempo_train_new.to_csv('../data/bea-base-train-flat/{}_train_5000.csv'.format(target), index=False)
    speech_tempo_dev_new.to_csv('../data/bea-base-train-flat/{}_dev_5000.csv'.format(target), index=False)