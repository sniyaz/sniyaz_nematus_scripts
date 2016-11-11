import numpy
import os
import sys

VOCAB_SIZE = 90000
SRC = "en"
TGT = "de"
DATA_DIR = "data/"

from nematus.nmt import train


if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                    reload_=True,
                    dim_word=500,
                    dim=1024,
                    n_words=None,
                    n_words_src=None,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=50,
                    batch_size=80,
                    valid_batch_size=80,
                    datasets=[DATA_DIR + '/corpus.bpe.' + SRC, DATA_DIR + '/corpus.bpe.' + TGT],
                    #valid_datasets=[DATA_DIR + '/newstest2015.bpe.' + SRC, DATA_DIR + '/newstest2015.bpe.' + TGT],

                    valid_datasets=None,
                    dictionaries=[DATA_DIR + '/corpus.bpe.' + SRC + '.json',DATA_DIR + '/corpus.bpe.' + TGT + '.json'],
                    #validFreq=100,
		    #validFreq=10000,
                    dispFreq=1000,
                    saveFreq=30000,
                    sampleFreq=10000,
                    use_dropout=False,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script=None)
    print validerr
