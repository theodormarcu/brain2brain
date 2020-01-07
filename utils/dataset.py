import glob
import numpy as np
import pandas as pd

from scipy.io import loadmat
from collections import Counter


def get_vocab(conv_dir,
              subject, 
              production=True,
              window_ms=2000,
              min_repititions=1,
              exclude_words=set(['sp', '{lg}', '{ns}'])):
    '''
    Build the vocabulary by reading the datums
    '''

    word2freq = Counter()
    #window_fs = int(window_ms / 1000 * fs)
    #half_window = window_fs // 2
    columns = 'word onset offset accuracy speaker'.split(' ')

    files = glob.glob(conv_dir + f'/NY{subject}*/misc/*datum_conversation_trimmed.txt')
    for filename in files:

        df = pd.read_csv(filename,
                         delimiter=' ',
                         header=None,
                         names=columns)

        df.word = df.word.str.lower() # Make lower case
        
        if production is True:
            df = df[df.speaker == 'Speaker1']
        elif production is False:
            df = df[df.speaker != 'Speaker1']

        #df = df[df.onset - half_window > 0]
        #df = df[df.onset + half_window <
            
        word2freq.update(word for word in df.word.tolist() if word not in exclude_words)

    print('# Conversations:', len(files))
    print('Vocabulary size:', len(word2freq))
    print('Total number of words:', sum(word2freq.values()))
    print(word2freq.most_common(10))

    # Prune based on number of repititions
    if min_repititions > 1:
        word2freq = {word:freq for word,freq in word2freq.items() if freq >= min_repititions}

        print('# Conversations:', len(files))
        print('Vocabulary size:', len(word2freq))
        print('Total number of words:', sum(word2freq.values()))


    return word2freq


def build_design(conv_dir,
                 subject,
                 example2label,
                 fs=512,
                 bin_ms=50,
                 shift_ms=0,
                 window_ms=2000,
                 delimiter=',',
                 electrodes=[10], # NOTE need a better default
                 production=True,
                 conversations=None,
                 handle_datum=None,
                 datum_suffix=None):
    '''
    Builds the design matrix given the following inputs:
    '''


    words = []
    labels = []
    signals = []

    bin_fs = int(bin_ms / 1000 * fs)
    shift_fs = int(shift_ms / 1000 * fs)
    window_fs = int(window_ms / 1000 * fs)
    half_window = window_fs // 2
    n_bins = len(range(-half_window, half_window, bin_fs))
    if datum_suffix is None:
        datum_suffix = 'conversation_trimmed'


    # Use specificied conversation if given, otherwise use all
    if conversations is None:
        conversations = glob.glob(conv_dir + f'/NY{subject}*') 
        if len(conversations) == 0:
            print('[ERROR] No conversations found')
            exit(1)


    # Read every conversation and build the examples
    for conversation in conversations:

        # Read datum
        # ----------
        datum_fn = glob.glob(conversation + '/misc/*datum_%s.txt' % datum_suffix)
        if len(datum_fn) == 0:
            print('File DNE: ', conversation + '/misc/*datum_%s.txt' % datum_suffix)
            continue


        examples = []    
        with open(datum_fn[0], 'r') as fin:
            for line in fin:
                parts = line.split(delimiter)

                word = parts[0].lower().strip()
                speaker = parts[4]

                # Filter based on mode. None would include everything
                if production is True and speaker != 'Speaker1':
                    continue
                elif production is False and speaker == 'Speaker1':
                    continue

                # Build example
                example = {
                    'word': word,
                    'onset': parts[1],
                    'offset': parts[2],
                    'accuracy': parts[3],
                    'speaker': speaker
                    }

                # Optionally perform additional processing
                if handle_datum is not None:
                    example = handle_datum(line, example)

                    # Skip examples that are rejected
                    if example is None:
                        continue

                examples.append(example)
        
        
        # Read signals
        # ------------
        ecogs = []
        for electrode in electrodes:
            mat_fn = glob.glob(conversation + f'/preprocessed/*_{electrode}.mat')
            if len(mat_fn) == 0:
                print(f'[WARNING] electrode {electrode} DNE in {conversation}')
                continue
            ecogs.append(loadmat(mat_fn[0])['p1st'].squeeze().astype(np.float32)) # float32
            
        if len(ecogs) == 0:
            print(f'Skipping bad conversation: {conversation}')
            continue

        ecogs = np.asarray(ecogs).T # (T, n_electrodes)
        assert ecogs.ndim == 2
        assert ecogs.shape[1] == len(electrodes)
        

        # Build examples
        # --------------
        for example in examples:
            onset = int(float(example['onset']))
            
            start = onset - half_window + shift_fs
            end = onset + half_window + shift_fs
            
            if start < 0 or start > ecogs.shape[0]: continue
            if end < 0 or end > ecogs.shape[0]: continue
            
            # Split the area around the onset into bins and average each one
            word_signal = np.zeros((n_bins, len(electrodes)), np.float32)
            for i, frame in enumerate(np.array_split(ecogs[start:end,:], n_bins, axis=0)):
                word_signal[i] = frame.mean(axis=0)

            words.append(example['word'])
            labels.append(example2label(example))
            signals.append(word_signal.squeeze()) # Note squeeze


    signals = np.array(signals)
    labels = np.array(labels)
    print('Shape of signals:', signals.shape)
    if signals.size == 0:
        print('[ERROR] Signals is empty')
        exit(1)

    return signals, labels, words


if __name__ == '__main__':
    
    proj_dir = '/mnt/bucket/labs/hasson/ariel/247'
    conv_dir = proj_dir + '/conversation_space/crude-conversations/Podcast/'
    signals, labels, words = build_design(conv_dir, 717, lambda x: x['word'], production=False)
    import pdb; pdb.set_trace()

