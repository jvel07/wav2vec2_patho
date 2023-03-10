import os

import numpy as np
import torch
from speechbrain.pretrained.interfaces import Pretrained
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

# import more_itertools as mit

from common import utils


# config = utils.load_config('recipes/config_demencia16k-225B.yml')


def extract_embeddings(dataset_list, feature_extractor, model, chunk_size, config):
    """Function to extract embeddings (convolutional features and hidden states) from a given wav2vec2 model

    :param chunk_size: int, Size of the chunks to take for each utterance.
    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """

    model_used = config['pretrained_model_details']['checkpoint_path'].split('/')[-2]
    path_embs = "{0}/convs/{1}_convs_wav2vec2".format(config['paths']['out_embeddings'], model_used)
    path_hiddens = "{0}/hiddens/{1}_hiddens_wav2vec2".format(config['paths']['out_embeddings'], model_used)
    os.makedirs(config['paths']['out_embeddings'], exist_ok=True)

    sampling_rate = config['sampling_rate']
    chunk_size = chunk_size
    frame_step = chunk_size * sampling_rate
    print("Feature extraction process started...")

    # Iterating datasets
    for index, dataset in enumerate(
            dataset_list):  # this for is just in case of the existence of 'dev', 'test' datasets
        list_convs = []
        list_hiddens = []

        # tot_input_values = feature_extractor(dataset['speech'], return_tensors="pt", padding=True,
        #                                      feature_size=1, sampling_rate=sampling_rate)

        # iterating utterances
        for i in range(3):  # 0, len(dataset)):
            list_current_utterance_convs = []
            list_current_utterance_hiddens = []

            # getting the i utterance
            utterance = dataset[i]["speech"]
            print("Processing utterance {}...".format(i))

            # iterating frames of the utterance
            for frame in range(0, len(utterance), frame_step):
                # chunking the utterance
                current_segment = utterance[frame:frame + frame_step]

                # padding when current_segment is smaller than frame_step
                # if len(current_segment) < frame_step:
                #     current_segment = list(mit.padded(current_segment, 0.00000000001, frame_step))
                # print(len(current_segment))

                # computing features for the segment
                input_values_segment = feature_extractor(current_segment, return_tensors="pt", padding=True,
                                                         feature_size=1, sampling_rate=sampling_rate)
                if len(current_segment) < sampling_rate / 25:
                    break

                # getting the outputs from the fine-tuned model
                with torch.no_grad():
                    outputs_segment = model(input_values_segment.input_values, input_values_segment.attention_mask)
                    # outputs_segment = model(torch.unsqueeze(tot_input_values.input_values[i][current_segment], dim=0),
                    #                         torch.unsqueeze(tot_input_values.attention_mask[i][current_segment], dim=0))

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


####################  ECAPA START ####################
class Encoder(Pretrained):
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings,
                torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings


def extract_ecapa_and_save(dataset_list, model, chunk_size, config):
    """Function to extract embeddings from a given ecapa model

    :param chunk_size: int, Size of the chunks to take for each utterance.
    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """
    checkpoint_path = config['pretrained_model_details']['checkpoint_path']

    if "jonatasgrosman" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "facebook" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "emotion" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    else:
        model_used = checkpoint_path.split('/')[-2]

    sampling_rate = config['sampling_rate']
    chunk_size = chunk_size
    frame_step = chunk_size * sampling_rate

    # Iterating datasets
    for index, dataset in enumerate(
            dataset_list):  # this for is just in case of the existence of 'dev', 'test' datasets
        list_convs = []
        list_hiddens = []

        print("Computing general features using Feature Extractor...")
        # tot_input_values = feature_extractor(dataset['speech'], return_tensors="pt", padding=True,
        #                                      feature_size=1, sampling_rate=sampling_rate)

        # iterating utterances
        if len(dataset) < 1:
            os.sys
        for i in range(0, len(dataset)):
            print("iteration started")
            list_current_utterance_convs = []
            list_current_utterance_hiddens = []

            # getting the i utterance
            utterance = dataset[i]["speech"]
            print("Processing utterance {}...".format(i))

            # iterating frames of the utterance
            for frame in range(0, len(utterance), frame_step):
                # chunking the utterance
                current_segment = utterance[frame:frame + frame_step]

                # padding when current_segment is smaller than frame_step
                # if len(current_segment) < frame_step:
                #     current_segment = list(mit.padded(current_segment, 0.00000000001, frame_step))
                # print(len(current_segment))

                # computing features for the segment
                input_values_segment = feature_extractor(current_segment, return_tensors="pt", padding=True,
                                                         feature_size=1, sampling_rate=sampling_rate)
                if len(current_segment) < sampling_rate / 25:
                    break

                # getting the outputs from the fine-tuned model
                with torch.no_grad():
                    outputs_segment = model(input_values_segment.input_values, input_values_segment.attention_mask)
                    # outputs_segment = model(torch.unsqueeze(tot_input_values.input_values[i][current_segment], dim=0),
                    #                         torch.unsqueeze(tot_input_values.attention_mask[i][current_segment], dim=0))

                # extract features from the last CNN layer
                segment_convs = outputs_segment.extract_features.detach().numpy()
                list_current_utterance_convs.append(segment_convs)

                # extract features corresponding to the sequence of last hidden states
                segment_hidden = outputs_segment.last_hidden_state.detach().numpy()
                list_current_utterance_hiddens.append(segment_hidden)

            # accumulating each wav into a list
            current_utterance_convs = np.concatenate(list_current_utterance_convs, axis=1)
            current_utterance_convs_pooled = np.squeeze(np.mean(current_utterance_convs, axis=1))
            # print(current_utterance_convs_pooled.shape)

            current_utterance_hiddens = np.concatenate(list_current_utterance_hiddens, axis=1)
            current_utterance_hidden_pooled = np.squeeze(np.mean(current_utterance_hiddens, axis=1))

            # defining paths and saving
            utterance_name = os.path.basename(dataset[i]['file']).split(".")[0]
            path_embs = config['paths']['out_embeddings'] + model_used
            os.makedirs(path_embs, exist_ok=True)
            file_convs = "{0}/convs_wav2vec2_{1}".format(path_embs, utterance_name)
            file_hiddens = "{0}/hiddens_wav2vec2_{1}".format(path_embs, utterance_name)
            np.save(file_convs, current_utterance_convs_pooled)
            np.save(file_hiddens, current_utterance_hidden_pooled)
            print(
                "Utterance embeddings (convs and hidden states) saved to {} and \n {}".format(file_convs, file_hiddens))
            # print("/n With shapes {}, and {}, respectively.".format(current_utterance_convs_pooled.shape, current_utterance_hidden_pooled.shape))


def extract_ecapa_original(dataset_list, model, config):
    """Function to extract embeddings (convolutional features and hidden states) from a given wav2vec2 model

    :param chunk_size: int, Size of the chunks to take for each utterance.
    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """
    checkpoint_path = config['pretrained_model_details']['checkpoint_path']

    if "jonatasgrosman" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "facebook" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "emotion" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "yangwang825" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    else:
        model_used = checkpoint_path.split('/')[-2]

    print("Feature extraction process started...")
    # Iterating datasets
    for index, dataset in enumerate(dataset_list):  # this for is just in case of the existence of 'dev', 'test' datasets
        list_embeddings = []
        # iterating utterances
        for i in range(len(dataset)):
            # getting the i utterance
            utterance = torch.tensor(dataset[i]["speech"])
            print("Processing utterance {}...".format(i))

            # extracting features
            embeddings = model.encode_batch(utterance)
            embeddings = torch.squeeze(embeddings)
            list_embeddings.append(embeddings.detach().numpy())

        # united array of all the wavs
        embs = np.asanyarray(np.vstack(list_embeddings))

        # defining paths and saving
        utterance_name = os.path.basename(dataset[i]['file']).split(".")[0]
        path_embs = config['paths']['out_embeddings'] + model_used
        os.makedirs(path_embs, exist_ok=True)
        file_embs = "{0}/embs_ecapa".format(path_embs)
        np.save(file_embs, embs)
        print("Utterance embeddings (convs and hidden states) saved to {}".format(file_embs))
        # print("/n With shapes {}, and {}, respectively.".format(current_utterance_convs_pooled.shape, current_utterance_hidden_pooled.shape))

######### ECAPA END   #########


def extract_embeddings_and_save(dataset_list, feature_extractor, model, chunk_size, config):
    """Function to extract embeddings (convolutional features and hidden states) from a given wav2vec2 model

    :param chunk_size: int, Size of the chunks to take for each utterance.
    :param model: Torch Wav2Vec2 pre-trained model (loaded).
    :param dataset_list: List. Use it for more than one set (e.g., dev, test).
    :param feature_extractor: Object. An instance of either Wav2Vec2Processor or Wav2Vec2FeatureExtractor.
    :return:
    """
    checkpoint_path = config['pretrained_model_details']['checkpoint_path']

    if "jonatasgrosman" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "facebook" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    elif "emotion" in checkpoint_path:
        model_used = checkpoint_path.split('/')[-1]
    else:
        model_used = checkpoint_path.split('/')[-2]

    sampling_rate = config['sampling_rate']
    chunk_size = chunk_size
    frame_step = chunk_size * sampling_rate

    # Iterating datasets
    for index, dataset in enumerate(dataset_list):  # this for is just in case of the existence of 'dev', 'test' datasets
        # print("Computing general features using Feature Extractor...")
        # tot_input_values = feature_extractor(dataset['speech'], return_tensors="pt", padding=True,
        #                                      feature_size=1, sampling_rate=sampling_rate)

        # iterating utterances
        if len(dataset) < 1:
            os.sys
        # for i in range(0, len(dataset)):
        for i in (pbar := tqdm(range(0, len(dataset)), desc="Extracting Embeddings", position=0)):
            list_current_utterance_convs = []
            list_current_utterance_hiddens = []

            # getting the i utterance
            utterance = dataset[i]["speech"]
            utterance_name = os.path.basename(dataset[i]['file']).split(".")[0]
            # print("Processing utterance {}...".format(i))
            pbar.set_description("Processing utterance {}".format(utterance_name))

            # iterating frames of the utterance
            for frame in range(0, len(utterance), frame_step):
                # chunking the utterance
                current_segment = utterance[frame:frame + frame_step]

                # padding when current_segment is smaller than frame_step
                # if len(current_segment) < frame_step:
                #     current_segment = list(mit.padded(current_segment, 0.00000000001, frame_step))
                # print(len(current_segment))

                # computing features for the segment
                input_values_segment = feature_extractor(current_segment, return_tensors="pt", padding=True,
                                                         feature_size=1, sampling_rate=sampling_rate)
                if len(current_segment) < sampling_rate / 25:
                    break

                # getting the outputs from the fine-tuned model
                with torch.no_grad():
                    outputs_segment = model(input_values_segment.input_values, input_values_segment.attention_mask)
                    # outputs_segment = model(torch.unsqueeze(tot_input_values.input_values[i][current_segment], dim=0),
                    #                         torch.unsqueeze(tot_input_values.attention_mask[i][current_segment], dim=0))

                # extract features from the last CNN layer
                # segment_convs = outputs_segment.extract_features.detach().numpy()
                # list_current_utterance_convs.append(segment_convs)

                # extract features corresponding to the sequence of last hidden states
                segment_hidden = outputs_segment.last_hidden_state.detach().numpy()
                list_current_utterance_hiddens.append(segment_hidden)

            # accumulating each wav into a list
            # current_utterance_convs = np.concatenate(list_current_utterance_convs, axis=1)
            # current_utterance_convs_pooled = np.squeeze(np.mean(current_utterance_convs, axis=1))
            # print(current_utterance_convs_pooled.shape)

            current_utterance_hiddens = np.concatenate(list_current_utterance_hiddens, axis=1)
            current_utterance_hidden_pooled = np.squeeze(np.mean(current_utterance_hiddens, axis=1))

            # defining paths and saving
            path_embs = config['paths']['out_embeddings'] + model_used
            os.makedirs(path_embs, exist_ok=True)
            file_convs = "{0}/convs_wav2vec2_{1}".format(path_embs, utterance_name)
            file_hiddens = "{0}/hiddens_wav2vec2_{1}".format(path_embs, utterance_name)
            # np.save(file_convs, current_utterance_convs_pooled)
            np.save(file_hiddens, current_utterance_hidden_pooled)
            # print("Utterance embeddings (convs and hidden states) saved to {} and \n {}".format(file_convs, file_hiddens))
            pbar.set_description("Embeddings saved to {}".format(file_hiddens))
            # print("/n With shapes {}, and {}, respectively.".format(current_utterance_convs_pooled.shape, current_utterance_hidden_pooled.shape))


def extract_embeddings_original(dataset_list, feature_extractor, model, config):
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
    print("Feature extraction process started...")

    # Iterating datasets
    for index, dataset in enumerate(
            dataset_list):  # this for is just in case of the existence of 'dev', 'test' datasets
        list_convs = []
        list_hiddens = []

        # tot_input_values = feature_extractor(dataset['speech'], return_tensors="pt", padding=True,
        #                                      feature_size=1, sampling_rate=sampling_rate)

        # iterating utterances
        for i in range(3):  # 0, len(dataset)):
            list_current_utterance_convs = []
            list_current_utterance_hiddens = []

            # getting the i utterance
            utterance = dataset[i]["speech"]
            print("Processing utterance {}...".format(i))

            # iterating frames of the utterance
            input_values_utterance = feature_extractor(utterance, return_tensors="pt", padding=True,
                                                       feature_size=1, sampling_rate=sampling_rate)

            # getting the outputs from the fine-tuned model
            with torch.no_grad():
                outputs_segment = model(input_values_utterance.input_values, input_values_utterance.attention_mask)
                # outputs_segment = model(torch.unsqueeze(tot_input_values.input_values[i][current_segment], dim=0),
                #                         torch.unsqueeze(tot_input_values.attention_mask[i][current_segment], dim=0))

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


def extract_embeddings_gabor(dataset_list, feature_extractor, model, chunk_size, config):
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
