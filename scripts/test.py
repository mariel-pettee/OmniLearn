import horovod.tensorflow.keras as hvd
import tensorflow as tf
import socket
import os

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)

print(f"[Rank {hvd.rank()}] Host: {socket.gethostname()}, "
      f"Local Rank: {hvd.local_rank()}, "
      f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")