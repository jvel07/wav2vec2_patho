import os

import numpy as np
import torch
import more_itertools as mit

from common import utils


config = utils.load_config('recipes/config_demencia16k-225B.yml')


def extract_embeddings(dataset_list, feature_extractor, model, chunk_size):
    """Function to extract embeddings (convolutional features and hidden states) from a given wav2vec2 model

    :param chunk_size: int, Size of the chunks to take for each utterance.
    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """

    model_used = config['pretrained_model_details']['checkpoint_path'].split('/')[-2]
    path_embs = "{0}/{1}_convs_wav2vec2".format(config['paths']['out_embeddings'], model_used)
    path_hiddens = "{0}/{1}_hiddens_wav2vec2".format(config['paths']['out_embeddings'], model_used)
    os.makedirs(config['paths']['out_embeddings'], exist_ok=True)

    sampling_rate = config['sampling_rate']
    chunk_size = chunk_size
    frame_step = chunk_size * sampling_rate
    print("Feature extraction process started...")

    # Iterating datasets
    for index, dataset in enumerate(dataset_list):  # this for is just in case of the existence of 'dev', 'test' datasets
        list_convs = []
        list_hiddens = []

        # iterating utterances
        for i in range(3):#0, len(dataset)):
            list_current_utterance_convs = []
            list_current_utterance_hiddens = []

            # getting the i utterance
            utterance = dataset[i]["speech"]
            print("Processing utterance {}...".format(i))

            for frame in range(0, len(utterance), frame_step):
                # chunking the utterance
                current_segment = utterance[frame:frame + frame_step]

                # padding when current_segment is smaller than frame_step
                if len(current_segment) < frame_step:
                    current_segment = list(mit.padded(current_segment, 0.00000000001, frame_step))

                # computing features for the segment
                input_values_segment = feature_extractor(current_segment, return_tensors="pt", padding=True,
                                                         feature_size=1, sampling_rate=sampling_rate)
                if len(current_segment) < sampling_rate / 25:
                    break

                # getting the outputs from the fine-tuned model
                with torch.no_grad():
                    outputs_segment = model(input_values_segment.input_values, input_values_segment.attention_mask)

                # extract features from the last CNN layer
                segment_convs = outputs_segment.extract_features.detach().numpy()
                list_current_utterance_convs.append(segment_convs)

                # extract features corresponding to the sequence of last hidden states
                segment_hidden = outputs_segment.last_hidden_state.detach().numpy()
                list_current_utterance_hiddens.append(segment_hidden)

            # accumulating each wav into a list
            current_utterance_convs = np.concatenate(list_current_utterance_convs, axis=1)
            print(current_utterance_convs.shape)
            list_convs.append(current_utterance_convs)

            current_utterance_hiddens = np.concatenate(list_current_utterance_hiddens, axis=1)
            list_hiddens.append(current_utterance_hiddens)

        # united array of all the wavs
        convs = np.asanyarray(np.vstack(list_convs))
        hiddens = np.asanyarray(np.vstack(list_hiddens))

        print("Final lengths:", len(list_convs), len(list_hiddens))

        np.save(path_embs, convs)
        np.save(path_hiddens, hiddens)
        print("Convolutional embeddings saved to {}. \n Hidden states saved to {}".format(path_embs, path_hiddens))
        print("/n With shapes {}, and {}, respectively.".format(convs.shape, hiddens.shape))


def extract_embeddings_2(dataset_list, feature_extractor, model, batch_size):
    """Function to extract embeddings (convolutional features and hidden states) from a given wav2vec2 model

    :param batch_size: int, Number of utterances to take.
    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """

    list_convs = []
    list_hidden = []
    path_embs = "{0}convs_wav2vec2".format(config['paths']['out_embeddings'], config['task'])
    path_hiddens = "{0}hiddens_wav2vec2".format(config['paths']['out_embeddings'], config['task'])

    for index, set_ in enumerate(dataset_list):
        tot_input_values = feature_extractor(set_['speech'], return_tensors="pt", padding=True,
                                             feature_size=1, sampling_rate=16000)
        for i in range(0, len(set_), batch_size):
            #     print(train_dataset[i:i+batch_size]['file_name'])
            #     files.append(train_dataset[i:i+batch_size]['file_name'])
            #     input_values = feature_extractor(train_dataset[i:i+batch_size]["speech"], return_tensors="pt", padding=True,
            #                                      feature_size=1, sampling_rate=16000 )#.input_values  # Batch size 1
            #     outputs = model(**tot_input_values[i:i+batch_size])
            # tot_input_values = feature_extractor(set_[i]["speech"], return_tensors="pt", padding=True,
            #                                      feature_size=1, sampling_rate=16000)
            # if tot_input_values are computed in the first "if"
            outputs = model(tot_input_values.input_values[i:i + batch_size], tot_input_values.attention_mask[i:i + batch_size])
            # if tot_input_values are computed per utterance
            # outputs = model(tot_input_values.input_values, tot_input_values.attention_mask)

            # extract features from the last CNN layer
            convs = outputs.extract_features.detach().numpy()
            print(convs.shape)
            # utils.save_data_iteratively(file_path=path_embs, data=convs)
            list_convs.append(convs)

            # extract features corresponding to the sequence of last hidden states
            hidden = outputs.last_hidden_state.detach().numpy()
            list_hidden.append(hidden)
            # utils.save_data_iteratively(file_path=path_embs, data=hidden)

            print("Processed:", i)

            embs = np.asanyarray(np.vstack(list_convs))
            hiddens = np.asanyarray(np.vstack(list_hidden))
            #
            # os.makedirs(config['paths']['out_embeddings'], exist_ok=True)
            np.save(path_embs, embs)
            np.save(path_hiddens, hiddens)
            print("Convolutional embeddings saved to {}. \n Hidden states saved to {}".format(path_embs, path_hiddens))
            print("/n With shapes {}, and {}, respectively.".format(convs.shape, hidden.shape))


def extract_embeddings_gabor(dataset_list, feature_extractor, model, chunk_size):
    """Function to extract embeddings (convolutional features and hidden states) from a given wav2vec2 model

    :param chunk_size: int, Size of the chunks to take for each utterance.
    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """
    path_embs = "{0}convs_wav2vec2".format(config['paths']['out_embeddings'], config['task'])
    path_hiddens = "{0}hiddens_wav2vec2".format(config['paths']['out_embeddings'], config['task'])
    dataset = dataset_list[0]
    for i in range(0, len(dataset)):
        utterance = dataset[i]["speech"]
        print('processing ' + str(i))
        samplingRate = 16000
        astep = chunk_size * samplingRate

        conv = np.empty(shape=[0, 0], dtype=np.float32)
        hidden = np.empty(shape=[0, 0], dtype=np.float32)

        for fi in range(0, len(utterance), astep):
            #        print(fi, ", ", len(speech[fi:fi+astep]))
            actsegment = utterance[fi:fi + astep]
            if len(actsegment) < samplingRate / 25:
                break
            # print(len(speech))

            tot_input_value = feature_extractor(actsegment, return_tensors="pt", padding=True,
                                                feature_size=1, sampling_rate=samplingRate)
            # print(tot_input_value.input_values.shape)

            with torch.no_grad():
                try:
                    inputs = tot_input_value.input_values
                    attentions = tot_input_value.attention_mask
                    outputs = model(inputs, attentions)
                except AttributeError:
                    inputs = tot_input_value.input_values
                    outputs = model(inputs)

            try:
                actconv = outputs.extract_features.detach()
                actconv = actconv[0]
                if torch.cuda.is_available():
                    actconv = actconv.cpu()
                actconv = actconv.numpy()
                #        print(actconv.shape)
                if (fi == 0):
                    conv = actconv
                else:
                    conv = np.concatenate((conv, actconv))
            except AttributeError:
                pass

            try:
                acthidden = outputs.last_hidden_state
                acthidden = acthidden[0]
                if torch.cuda.is_available():
                    acthidden = acthidden.cpu()
                acthidden = acthidden.numpy()
                if (fi == 0):
                    hidden = acthidden
                else:
                    hidden = np.concatenate((hidden, acthidden))
            except AttributeError:
                pass

        print(conv.shape)
        np.save(path_embs, conv)
        np.save(path_hiddens, hidden)
