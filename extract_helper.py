import numpy as np

from common import utils

config = utils.load_config('recipes/config_demencia16k-225B.yml')


def extract_embeddings(dataset_list, feature_extractor, model):
    """Function to extract embeddings (convolutional features and hidden states) from a given wav2vec2 model

    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """

    # list_convs = []
    # list_hidden = []
    sampling_rate = config['sampling_rate']
    chunk_size = 60
    frame_step = chunk_size * sampling_rate
    print("Feature extraction process started...")
    for index, dataset in enumerate(dataset_list):
        # print("Feature extractor computed finished with processing the dataset...")
        # print("Now computing the embeddings...")
        #
        conv_array = np.empty(shape=[0, 0, 0], dtype=np.float32)
        hidden_array = np.empty(shape=[0, 0, 0], dtype=np.float32)
        for i in range(0, len(dataset)):  # this for is just in case of the existence of 'dev', 'test' datasets
            utterance = dataset[i]["speech"]
            for frame in range(0, len(utterance), frame_step):
                current_segment = utterance[frame:frame+frame_step]
                if len(current_segment) < sampling_rate / 25:
                    break
                tot_input_values = feature_extractor(current_segment, return_tensors="pt", padding=True,
                                                     feature_size=1, sampling_rate=sampling_rate)
            #     outputs = model(**tot_input_values[i:i+batch_size])
                outputs = model(tot_input_values.input_values, tot_input_values.attention_mask)

                # extract features from the last CNN layer
                current_convs = outputs.extract_features.detach().numpy()
                if frame == 0:
                    # print(conv_array.shape, current_convs.shape)
                    conv_array = current_convs
                else:
                    # print(conv_array.shape, current_convs.shape)
                    conv_array = np.concatenate((conv_array, current_convs), axis=1)
                # list_convs.append(convs)

                # extract features corresponding to the sequence of last hidden states
                current_hidden = outputs.last_hidden_state.detach().numpy()
                if frame == 0:
                    hidden_array = current_hidden
                else:
                    hidden_array = np.concatenate((hidden_array, current_hidden), axis=1)
                # list_hidden.append(hidden)

            print("Processed {}%...".format(i))

        embs = np.asanyarray(np.vstack(conv_array))
        hiddens = np.asanyarray(np.vstack(hidden_array))
        path_embs = "{0}/convs_wav2vec2".format(config['paths']['out_embeddings'], config['task'])
        path_hiddens = "{0}/hiddens_wav2vec2".format(config['paths']['out_embeddings'], config['task'])
        np.save(path_embs, embs)
        np.save(path_hiddens, hiddens)
        print("Convolutional embeddings saved to {}. \n Hidden states saved to {}".format(path_embs, path_hiddens))
        print("/n With shapes {}, and {}, respectively.".format(embs.shape, hiddens.shape))
