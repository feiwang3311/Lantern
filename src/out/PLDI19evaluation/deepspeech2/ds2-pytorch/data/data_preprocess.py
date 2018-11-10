## Xilun: Need modify this file
## copied from data_loader.py

import os
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import scipy.signal
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import time
import json
import argparse
import sys
sys.path.append('../')
import pytorch.params as params
import struct

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def load_audio(path):
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        self.paths = path is not None and librosa.util.find_files(path)
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_src = load_audio(noise_path)
        noise_offset_fraction = np.random.rand()
        noise_dst = np.zeros_like(data)
        src_offset = int(len(noise_src) * noise_offset_fraction)
        src_left = len(noise_src) - src_offset
        dst_offset = 0
        dst_left = len(data)
        while dst_left > 0:
            copy_size = min(dst_left, src_left)
            np.copyto(noise_dst[dst_offset:dst_offset + copy_size],
                      noise_src[src_offset:src_offset + copy_size])
            if src_left > dst_left:
                dst_left = 0
            else:
                dst_left -= copy_size
                dst_offset += copy_size
                src_left = len(noise_src)
                src_offset = 0
        data += noise_level * noise_dst
        return data


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size

class SpectrogramAndPathDataset(SpectrogramDataset):
    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript, audio_path

class SpectrogramAndLogitsDataset(SpectrogramDataset):
    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        logit_path = os.path.join(
        os.path.split(os.path.split(audio_path)[0])[0],
            "logits",
            os.path.splitext(os.path.split(audio_path)[1])[0] + ".pth"
        )
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        logits = torch.load(logit_path)
        return spect, transcript, audio_path, logits

def _collate_fn_logits(batch):
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)

    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []

    paths = []

    longest_logit = max(batch, key=lambda p: p[3].size(0))[3]
    logit_len = longest_logit.size(0)
    nclasses = batch[0][3].size(-1)
    logits = torch.FloatTensor(minibatch_size, logit_len, nclasses)
    logits.fill_(float('-inf'))

    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        paths.append(sample[2])
        logit = sample[3]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
        logits[x,:logit.size(0)].copy_(logit)

    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes, paths, logits

def _collate_fn_paths(batch):
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    paths = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        paths.append(sample[2])
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes, paths, None

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

class AudioDataAndLogitsLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataAndLogitsLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_logits

class AudioDataAndPathsLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataAndPathsLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_paths

def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                            augmented_filename,
                                                                            " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio

## begin of data processor
def to_np(x):
    return x.data.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def write_dataset_to_binary_file(filename, data_loader):
    start_iter = 0 
    for i, (data) in enumerate(data_loader, start=start_iter):
        if i == len(data_loader):
            break
        inputs, targets, input_percentages, target_sizes = data
        if i == start_iter:
            with open(filename, 'wb') as f:
                f.write(struct.pack("@i", params.batch_size))
                f.write(struct.pack("@i", len(data_loader)-start_iter))
                # bin format for Lantern:
                # Int: batch_size, num_batches
                # Int: freq_size, max_length
                # Tensor: inputs(FloatTensor[batch_size, 1, freq_size, max_length])
                # Tensor: input_percentages(FloatTensor[batch_size])
                # Tensor: target_sizes(IntTensor[batch_size])
                # Tensor: targets(IntTensor[sum(target_sizes)]) -- each target is a character label
        with open(filename, 'ab') as f:
            f.write(struct.pack("@i", inputs[0][0].size(0)))
            f.write(struct.pack("@i", inputs[0][0].size(1)))
            # writing inputs
            for by in inputs.storage().tolist():
                f.write(struct.pack("@f", by))
                # writing input_percentages
            for by in input_percentages.storage().tolist():
                f.write(struct.pack("@f", by))
                # writing target_sizes
            for by in target_sizes.storage().tolist():
                f.write(struct.pack("@i", by))
                # writing targets
            for by in targets.storage().tolist():
                f.write(struct.pack("@i", by))
                
def verify_binary_file(filename, data_loader):                
    # Verify: read the content out of the bin file and compare with original tensor
    struct_head_fmt = '@ii'
    struct_head_len = struct.calcsize(struct_head_fmt)
    struct_head_unpack = struct.Struct(struct_head_fmt).unpack_from
    struct_batch_head_fmt = struct_head_fmt
    struct_batch_head_len = struct.calcsize(struct_batch_head_fmt)
    struct_batch_head_unpack = struct.Struct(struct_batch_head_fmt).unpack_from
    # inputs, input_percentages, target_sizes
    struct_batch_fmt_format1 = '@{}f{}f{}i'
    # targets
    struct_batch_fmt_format2 = '@{}i'

    with open(filename, 'rb') as f:
        data = f.read(struct_head_len)
        if not data:
            print("failure in loading head\n")
            raise
        batch_size, num_batches = struct_head_unpack(data)
        print('batch_size = {}, num_batches = {}'.format(batch_size, num_batches))

        start_iter = 0
        for i, (origin_data) in enumerate(data_loader, start=start_iter):
            if i == len(data_loader):
                break
            ori_inputs, ori_targets, ori_input_percentages, ori_target_sizes = origin_data
            print('spectrogram shape = ', ori_inputs.size())
            # get next batch
            data = f.read(struct_batch_head_len)
            if not data:
                print("failure in loading batch head\n")
                raise

            freq_size, max_seqlength = struct_batch_head_unpack(data)
            print(freq_size, max_seqlength)
            input_size = batch_size * 1 * freq_size * max_seqlength
            struct_batch_fmt1 = struct_batch_fmt_format1.format(input_size, batch_size, batch_size)
            struct_batch_fmt1_len = struct.calcsize(struct_batch_fmt1)

            # print(struct_batch_fmt1, struct_batch_fmt1_len, '\n')
            struct_batch_fmt1_unpack = struct.Struct(struct_batch_fmt1).unpack_from

            data = f.read(struct_batch_fmt1_len)
            if not data:
                print("failure in loading batch part1\n")
                raise

            # print(len(data), struct_batch_fmt1_len)
            unpacked_data = struct_batch_fmt1_unpack(data)
            inputs = list(unpacked_data[:input_size])
            input_percentages = list(unpacked_data[input_size: input_size+batch_size])
            target_sizes = list(unpacked_data[input_size+batch_size: input_size+2*batch_size])

            # print(target_sizes, '\n')
            # it works
            inputs_tensor = torch.tensor(
                np.ndarray((batch_size, 1, freq_size, max_seqlength), buffer=np.asarray(inputs)),
                dtype=torch.float)

            input_percentages_tensor = torch.tensor(
                np.ndarray((batch_size), buffer=np.asarray(input_percentages)),
                dtype=torch.float)


            target_sizes_tensor = torch.tensor(
                np.ndarray((batch_size), buffer=np.asarray(target_sizes), dtype=np.int),
                dtype=torch.int)
            # print(np.ndarray((batch_size), buffer=np.asarray(target_sizes), dtype=np.int))
            # print(type(inputs_tensor), type(ori_inputs), '\n')
            # print(inputs_tensor.size(), ori_inputs.size(), '\n')
            # print(ori_input_percentages[0].eq(torch.Tensor([input_percentages[0]])))
            # print(ori_input_percentages[0].eq(input_percentages_tensor[0]))
            print(inputs_tensor, ori_inputs)
            print(torch.eq(inputs_tensor, ori_inputs))
            # print(torch.eq(input_percentages_tensor, ori_input_percentages))
            # print(input_percentages_tensor, ori_input_percentages)
            assert inputs_tensor.equal(ori_inputs), "inputs not equal"
            assert input_percentages_tensor.equal(ori_input_percentages), "input_percentage not equal"
            # print(target_sizes_tensor, ori_target_sizes)
            assert target_sizes_tensor.equal(ori_target_sizes), "target_sizes not equal"

            target_size = sum(target_sizes)
            struct_batch_fmt2 = struct_batch_fmt_format2.format(target_size)
            struct_batch_fmt2_len = struct.calcsize(struct_batch_fmt2)
            struct_batch_fmt2_unpack = struct.Struct(struct_batch_fmt2).unpack_from

            data = f.read(struct_batch_fmt2_len)
            # print(len(data), struct_batch_fmt1_len)
            if not data:
                print("failure in loading batch part2\n")
                raise
            targets = list(struct_batch_fmt2_unpack(data))
            targets_tensor = torch.tensor(
                np.ndarray((target_size), buffer=np.asarray(targets), dtype=np.int),
                dtype=torch.int
            )
            assert targets_tensor.equal(ori_targets), "targets not equal"
    
def main():
    WRITE_DATASET = True
    VERIFY_DATASET = True
    parser = argparse.ArgumentParser(description='DeepSpeech data preprocessing')
    args = parser.parse_args()
    print("=======================================================")
    for arg in vars(args):
      print("***%s = %s " %  (arg.ljust(25), getattr(args, arg)))
    print("=======================================================")

    save_folder = './test/'
    
    try:
        os.makedirs(save_folder)
    except:
        pass

#    dataset_name = 'test'
    for dataset_name in ['train', 'val', 'test']:
        filename = save_folder + 'deepspeech_{}.bin'.format(dataset_name)
        manifest = {'train': params.train_manifest, 'val': params.val_manifest, 'test': params.test_manifest}
        manifest_filepath = manifest[dataset_name]
        if manifest_filepath is None:
            raise
        print(manifest_filepath)

        with open(params.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))
        audio_conf = dict(sample_rate=params.sample_rate,
                          window_size=params.window_size,
                          window_stride=params.window_stride,
                          window=params.window,
                          noise_dir=params.noise_dir,
                          noise_prob=params.noise_prob,
                          noise_levels=(params.noise_min, params.noise_max))    
        dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=manifest_filepath,
                                     labels=labels, normalize=True, augment=params.augment)
        loader = AudioDataLoader(dataset, batch_size=params.batch_size,
                                 num_workers=1, drop_last=True)


        write_dataset_to_binary_file(filename, data_loader=loader)
#    verify_binary_file(filename, data_loader=loader)

if __name__ == '__main__':
    main()

