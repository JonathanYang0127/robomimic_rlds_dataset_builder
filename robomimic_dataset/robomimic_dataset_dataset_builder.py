from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import cv2
import h5py
import json
import io
from collections import defaultdict
import random
from copy import deepcopy
from PIL import Image
from collections import OrderedDict
import torch
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory


class RobomimicDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        if 'feature_metadata' not in dir(self):
            self.feature_metadata = self._get_feature_metadata()

    def _get_feature_metadata(self):
        config_path = '/iris/u/jyang27/ds_square_ac_keys_abs.json'
        ext_cfg = json.load(open(config_path, 'r'))
        self.config = config_factory(ext_cfg["algo_name"])
        with self.config.values_unlocked():
            self.config.update(ext_cfg)
        eval_dataset_cfg = self.config.train.data[0]
        dataset_path = os.path.expanduser(eval_dataset_cfg["path"])
        ds_format = self.config.train.data_format
        ObsUtils.initialize_obs_utils_with_config(self.config)
        self.shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=self.config.train.action_keys,
            all_obs_keys=self.config.all_obs_keys,
            ds_format=ds_format,
            verbose=True
        )
        self.feature_metadata = self._get_metadata_from_path(dataset_path)
        self.obs_keys = self.config.all_obs_keys
        self.action_keys = self.config.train.action_keys


    def _get_metadata_from_path(self, dataset_path):
        dataset_path = os.path.expanduser(dataset_path)
        f = h5py.File(dataset_path, "r")
        demo_id = list(f["data"].keys())[0]
        demo = f["data/{}".format(demo_id)]
        
        feature_meta = {}
        all_shapes = OrderedDict()
        all_dtypes = OrderedDict()

        action_keys = self.config.train.action_keys
        for key in action_keys:
            assert len(demo[key].shape) == 2
        action_dim = [sum([demo[key].shape[1] for key in action_keys])]
        feature_meta["ac_dim"] = action_dim

        all_obs_keys = [f"obs/{k}" for k in self.config.all_obs_keys]
        for k in sorted(all_obs_keys):
            data = demo[k]
            data_dtype = data[:].dtype

            initial_shape = data.shape[1:]
            all_shapes[k] = ObsUtils.get_processed_shape(
                obs_modality=ObsUtils.OBS_KEYS_TO_MODALITIES[k],
                input_shape=initial_shape,
            )
            all_dtypes[k] = data_dtype

        action_keys = self.config.train.action_keys
        for k in sorted(action_keys):
            data = demo[k]
            all_dtypes[k] = data[:].dtype
            all_shapes[k] = data.shape[1:]

        feature_meta['all_shapes'] = all_shapes
        feature_meta['all_obs_keys'] = all_obs_keys
        feature_meta['action_keys'] = action_keys
        feature_meta['all_dtypes'] = all_dtypes
        feature_meta['use_images'] = ObsUtils.has_modality("rgb", all_obs_keys)

        return feature_meta

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        if 'feature_metadata' not in dir(self):
            self._get_feature_metadata()
        
        obs_features = dict()
        for obs_key in self.feature_metadata['all_obs_keys']:
            nested_key = obs_key.split('/')[-1]
            if 'image' in obs_key:
                obs_features[nested_key] = tfds.features.Image(
                    shape=tuple(self.feature_metadata['all_shapes'][obs_key]),
                    dtype=self.feature_metadata['all_dtypes'][obs_key],
                )
            else:
                obs_features[nested_key] = tfds.features.Tensor(
                    shape=tuple(self.feature_metadata['all_shapes'][obs_key]),
                    dtype=self.feature_metadata['all_dtypes'][obs_key],
                )
        action_features = dict()
        for action_key in self.feature_metadata['action_keys']:
            nested_key = action_key.split('/')[-1]
            action_features[nested_key] = tfds.features.Tensor(
                shape=tuple(self.feature_metadata['all_shapes'][action_key]),
                dtype=self.feature_metadata['all_dtypes'][action_key],
            )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict(obs_features),
                    'action_dict': tfds.features.FeaturesDict(action_features),
                    'action': tfds.features.Tensor(
                        shape=tuple(self.feature_metadata['ac_dim']),
                        dtype=np.float32,
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_index': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'env_metadata': tfds.features.FeaturesDict({ 
                        'env_name': tfds.features.Text(
                            doc='Name of the enviroment'
                        ),
                        'env_version': tfds.features.Text(
                            doc='Version of the environment'
                        ),
                        'type': tfds.features.Scalar(
                            dtype=np.int32,
                            doc='Env version'
                        ),
                        'env_kwargs': tfds.features.Text(
                            doc='String of rest of the environment kwargs'
                        ),
                    }),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/afs/cs.stanford.edu/u/jyang27/tmp/autogen_configs/ril/diffusion_policy/square/ld/09-29-None/09-29-23-16-03-58/json/ds_square_ac_keys_abs.json'),
            #'val': self._generate_examples(''),
        }


    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _resize_and_encode(image, size):
            image = Image.fromarray(image)
            return np.array(image.resize(size, resample=Image.BICUBIC))
        

        def _parse_example(trainset_index):
            data = trainset[trainset_index]
             
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(len(data['actions'])):
                observation_dict = {}
                action_dict = {}
                for obs_key in self.feature_metadata['all_obs_keys']:
                    nested_key = obs_key.split('/')[-1]
                    observation_dict[nested_key] = data['obs'][nested_key][i]
                for action_key in self.feature_metadata['action_keys']:
                    nested_key = action_key.split('/')[-1]
                    action_dict[nested_key] = data[action_key][i]
                action = data['actions'][i]
                language_instruction = 'Execute a task.'

                # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()
                episode.append({
                    'observation': observation_dict,
                    'action_dict': action_dict,
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })
            
            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'episode_index': str(trainset_index),
                    'env_metadata': {
                        'env_name': env_metadata['env_name'],
                        'env_version': env_metadata['env_version'],
                        'type': env_metadata['type'],
                        'env_kwargs': str(env_metadata['env_kwargs'])
                    }
                }
            }
            # if you want to skip an example for whatever reason, simply return None
            return str(trainset_index), sample


        #Load metadata
        eval_dataset_cfg = self.config.train.data[0]
        dataset_path = os.path.expanduser(eval_dataset_cfg["path"])
        ds_format = self.config.train.data_format
        print(ds_format)
        env_metadata = FileUtils.get_env_metadata_from_dataset(
            dataset_path=dataset_path, ds_format=ds_format)
        #Load config
        trainset, validset = TrainUtils.load_data_for_training(
            self.config, obs_keys=self.config.all_obs_keys)  

        # for smallish datasets, use single-thread parsing
        for i in range(10): #len(trainset)
            yield _parse_example(i)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        #beam = tfds.core.lazy_imports.apache_beam
        #return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        #)


