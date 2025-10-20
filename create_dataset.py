import os, sys
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
import re


ROOT = Path(__file__).resolve().parent.parent
SDK_PATH = ROOT / 'CMU-MultimodalSDK'
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))
from mmsdk import mmdatasdk as md


def to_pickle(obj, path):
    with open(path, 'wb') as f: pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f: return pickle.load(f)

word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']
def return_unk(): return UNK


class CH_SIMS:
    def __init__(self, config):
        DATA_PATH = config.data_dir / 'unaligned.pkl'
        data = load_pickle(DATA_PATH)
        
        self.word2id = None
        self.pretrained_emb = None
        
        mode = config.mode
        if mode == 'dev': mode = 'valid'
            
        self.data = self._convert_split(data[mode])
        
        if len(self.data) > 0:
            _, v0, a0, _ = self.data[0][0]
            config.visual_size = v0.shape[1]
            config.acoustic_size = a0.shape[1]
        config.embedding_size = 0

    def _convert_split(self, split_data):
        samples = []
        num_samples = len(split_data['vision'])
        
        for i in range(num_samples):
            visual = split_data['vision'][i].astype(np.float32)
            acoustic = split_data['audio'][i].astype(np.float32)
            vid = split_data['id'][i]
            sentence_text = str(split_data['raw_text'][i])
            label_value = split_data['regression_labels'][i]
            label = np.array([label_value]).astype(np.float32)
            glove_ids = []
            actual_words = sentence_text.split()
            samples.append(((glove_ids, visual, acoustic, actual_words), label, vid))
        return samples

    def get_data(self, mode):
        return self.data, self.word2id, self.pretrained_emb

class MOSI:
    def __init__(self, config):
        DATA_PATH = config.data_dir
        try:
            self.train = load_pickle(DATA_PATH / 'train.pkl')
            self.dev = load_pickle(DATA_PATH / 'dev.pkl')
            self.test = load_pickle(DATA_PATH / 'test.pkl')
        except FileNotFoundError:
            self._create_data(config)

    def _create_data(self, config):
        DATA_PATH = config.data_dir
        if not DATA_PATH.exists(): DATA_PATH.mkdir(parents=True)
        
        DATASET = md.cmu_mosi
        md.mmdataset(DATASET.highlevel, DATA_PATH)
        md.mmdataset(DATASET.labels, DATA_PATH)
        
        features = ['CMU_MOSI_TimestampedWords', 'CMU_MOSI_VisualFacet_4.1', 'CMU_MOSI_COVAREP']
        recipe = {feat: DATA_PATH / f'{feat}.csd' for feat in features}
        dataset = md.mmdataset(recipe)

        def avg(intervals, features): return np.average(features, axis=0)
        dataset.align(features[0], collapse_functions=[avg])

        label_field = 'CMU_MOSI_Opinion_Labels'
        label_recipe = {label_field: DATA_PATH / f'{label_field}.csd'}
        dataset.add_computational_sequences(label_recipe, destination=None)
        dataset.align(label_field)
        
        self.train, self.dev, self.test = self._process_splits(dataset, DATASET.standard_folds)
        to_pickle(self.train, DATA_PATH / 'train.pkl')
        to_pickle(self.dev, DATA_PATH / 'dev.pkl')
        to_pickle(self.test, DATA_PATH / 'test.pkl')

    def _process_splits(self, dataset, splits):
        train_split, dev_split, test_split = splits.standard_train_fold, splits.standard_valid_fold, splits.standard_test_fold
        data_splits = {'train': [], 'dev': [], 'test': []}
        pattern = re.compile('(.*)\[.*\]')
        
        for segment in tqdm(dataset['CMU_MOSI_Opinion_Labels'].keys(), desc="Processing MOSI"):
            vid = re.search(pattern, segment).group(1)
            label = dataset['CMU_MOSI_Opinion_Labels'][segment]['features']
            _words = dataset['CMU_MOSI_TimestampedWords'][segment]['features']
            _visual = dataset['CMU_MOSI_VisualFacet_4.1'][segment]['features']
            _acoustic = dataset['CMU_MOSI_COVAREP'][segment]['features']

            if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]: continue

            actual_words, visual, acoustic = [], [], []
            for i, word in enumerate(_words):
                if word[0] != b'sp':
                    actual_words.append(word[0].decode('utf-8'))
                    visual.append(_visual[i, :])
                    acoustic.append(_acoustic[i, :])
            
            visual, acoustic = np.asarray(visual), np.asarray(acoustic)
            label = np.nan_to_num(label)
            visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (1e-6 + np.std(visual, axis=0, keepdims=True)))
            acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (1e-6 + np.std(acoustic, axis=0, keepdims=True)))
            
            data_point = ((None, visual, acoustic, actual_words), label, vid)
            
            if vid in train_split: data_splits['train'].append(data_point)
            elif vid in dev_split: data_splits['dev'].append(data_point)
            elif vid in test_split: data_splits['test'].append(data_point)
            
        return data_splits['train'], data_splits['dev'], data_splits['test']

    def get_data(self, mode):
        data = getattr(self, mode)
        return data, None, None

class MOSEI(MOSI):
    def __init__(self, config):
        super().__init__(config)

    def _create_data(self, config):
        DATA_PATH = config.data_dir
        if not DATA_PATH.exists(): DATA_PATH.mkdir(parents=True)
        
        DATASET = md.cmu_mosei
        md.mmdataset(DATASET.highlevel, DATA_PATH)
        md.mmdataset(DATASET.labels, DATA_PATH)
        
        features = ['CMU_MOSEI_TimestampedWords', 'CMU_MOSEI_VisualFacet42', 'CMU_MOSEI_COVAREP']
        recipe = {feat: DATA_PATH / f'{feat}.csd' for feat in features}
        dataset = md.mmdataset(recipe)

        def avg(intervals, features): return np.average(features, axis=0)
        dataset.align(features[0], collapse_functions=[avg])

        label_field = 'CMU_MOSEI_LabelsSentiment'
        label_recipe = {label_field: DATA_PATH / f'{label_field}.csd'}
        dataset.add_computational_sequences(label_recipe, destination=None)
        dataset.align(label_field)
        
        self.train, self.dev, self.test = self._process_splits(dataset, DATASET.standard_folds)
        to_pickle(self.train, DATA_PATH / 'train.pkl')
        to_pickle(self.dev, DATA_PATH / 'dev.pkl')
        to_pickle(self.test, DATA_PATH / 'test.pkl')

    def _process_splits(self, dataset, splits):
        # Override with MOSEI-specific field names
        train_split, dev_split, test_split = splits.standard_train_fold, splits.standard_valid_fold, splits.standard_test_fold
        data_splits = {'train': [], 'dev': [], 'test': []}
        pattern = re.compile('(.*)\[.*\]')
        
        for segment in tqdm(dataset['CMU_MOSEI_LabelsSentiment'].keys(), desc="Processing MOSEI"):
            vid = re.search(pattern, segment).group(1)
            label = dataset['CMU_MOSEI_LabelsSentiment'][segment]['features']
            _words = dataset['CMU_MOSEI_TimestampedWords'][segment]['features']
            _visual = dataset['CMU_MOSEI_VisualFacet42'][segment]['features']
            _acoustic = dataset['CMU_MOSEI_COVAREP'][segment]['features']

            if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]: continue

            actual_words, visual, acoustic = [], [], []
            for i, word in enumerate(_words):
                if word[0] != b'sp':
                    actual_words.append(word[0].decode('utf-8'))
                    visual.append(_visual[i, :])
                    acoustic.append(_acoustic[i, :])
            
            visual, acoustic = np.asarray(visual), np.asarray(acoustic)
            label = np.nan_to_num(label)
            visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (1e-6 + np.std(visual, axis=0, keepdims=True)))
            acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (1e-6 + np.std(acoustic, axis=0, keepdims=True)))
            
            data_point = ((None, visual, acoustic, actual_words), label, vid)
            
            if vid in train_split: data_splits['train'].append(data_point)
            elif vid in dev_split: data_splits['dev'].append(data_point)
            elif vid in test_split: data_splits['test'].append(data_point)
            
        return data_splits['train'], data_splits['dev'], data_splits['test']