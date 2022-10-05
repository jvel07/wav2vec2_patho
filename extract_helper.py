import numpy as np


def extract_embeddings(dataset_list, feature_extractor, batch_size, model):
    """Function to extract embeddings (convolutional features and hidden states) from a given wav2vec2 model

    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param batch_size: Int, portion of the dataset rows to compute per iteration.
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """

    list_convs = []
    list_hidden = []
    print("Feature extraction process started...")
    for index, dataset in enumerate(dataset_list):

        #     if index == 0:
        #         set_ = 'train'
        # if index == 0:
        #     set_ = 'dev'
        # else:
        #     set_ = 'test.txt'
        # set_='test.txt'

        tot_input_values = feature_extractor(dataset['speech'], return_tensors="pt", padding=True,
                                             feature_size=1, sampling_rate=16000)
        # tot_input_values.to(device)
        print("Feature extractor computed finished with processing the dataset...")
        print("Now computing the embeddings...")
        #
        for i in range(0, len(dataset), batch_size):
            #     print(train_dataset[i:i+batch_size]['file_name'])
            #     files.append(train_dataset[i:i+batch_size]['file_name'])
            #     input_values = feature_extractor(train_dataset[i:i+batch_size]["speech"], return_tensors="pt", padding=True,
            #                                      feature_size=1, sampling_rate=16000 )#.input_values  # Batch size 1
            #     outputs = model(**tot_input_values[i:i+batch_size])
            outputs = model(tot_input_values.input_values[i:i + batch_size],
                            tot_input_values.attention_mask[i:i + batch_size])

            # extract features from the last CNN layer
            convs = outputs.extract_features.detach().numpy()
            list_convs.append(convs)

            # extract features corresponding to the sequence of last hidden states
            hidden = outputs.last_hidden_state.detach().numpy()
            list_hidden.append(hidden)

            print(batch_size + i)

        embs = np.asanyarray(np.vstack(list_convs))
        hiddens = np.asanyarray(np.vstack(list_hidden))
        path_embs = "../data/{0}/embeddings/{1}_convs_wav2vec2".format(extract.task)
        path_hiddens = "../data/{0}/embeddings/{1}_hiddens_wav2vec2".format(extract.task)
        np.save(path_embs, embs)
        np.save(path_hiddens, hiddens)
        print("Convolutional embeddings saved to {}. \n Hidden states saved to {}".format(path_embs, path_hiddens))
