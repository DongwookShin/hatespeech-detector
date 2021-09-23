import os, sys, re, itertools, glob, time, random
import numpy as np
import tensorflow as tf

class ModelPeriodicCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, freq, ckpt_path):
        super().__init__()
        self.freq = freq
        self.ckpt_path = ckpt_path

    def on_epoch_begin(self, epoch, logs=None):
        if self.freq > 0:
            if epoch % self.freq == 0:
                os.makedirs((self.ckpt_path), exist_ok=True)
                fpath = '{}/e{:>05d}.h5'.format(self.ckpt_path, epoch)
                self.model.save_weights(fpath, save_format='h5')
                running_path = '{}/model.h5'.format(self.ckpt_path)
                if os.path.exists(running_path):
                    os.unlink(running_path)
                time.sleep(10)
                os.symlink(os.path.basename(fpath), running_path)

    def on_train_end(self, logs=None):
        os.makedirs((self.ckpt_path), exist_ok=True)
        fpath = '{}/model.h5'.format(self.ckpt_path)
        self.model.save_weights(fpath, save_format='h5')

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))