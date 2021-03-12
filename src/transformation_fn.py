from functools import partial

import tensorflow as tf
import constants
import nn_utils
# import transformation_fn

def roll(matrix, shift):
  # Assuming it to be a 3d-tensor of shape (batch, seq_len, seq_len)
  if shift > 0: #which means it's rightward shifting
    shift = tf.abs(shift)
    return tf.concat([ matrix[:, shift:, :], tf.zeros_like(matrix)[:, :shift,  :]], axis = 1) # In implementation it's actually up-rolling
  elif shift < 0:
    shift = tf.abs(shift)
    return tf.concat([matrix[:, :, shift:], tf.zeros_like(matrix)[:, :, :shift]], axis= 2)
  else:
    return matrix

def get_decedent_mtx(heads):
  # heads: (B, S)
  heads = tf.cast(heads, tf.int32)
  seq_len = tf.shape(heads)[1]
  print(seq_len)
  array_idx = tf.range(400, dtype=tf.int32)[:seq_len]
  mtx_idxer = tf.tile(tf.reshape(array_idx, [-1, 1]), [1, seq_len])
  idxer = tf.tile(tf.reshape(array_idx, [1, -1]), [seq_len, 1])
  heads = tf.tile(tf.expand_dims(heads, 1), [1, seq_len, 1])
  one_mtx = tf.ones_like(mtx_idxer, dtype=tf.float32)
  decedent_mtx = tf.map_fn(lambda x: tf.where(tf.logical_or(tf.equal(mtx_idxer, x), tf.equal(mtx_idxer, idxer)),  constants.VERY_LARGE * one_mtx, constants.VERY_SMALL * one_mtx), heads, dtype=tf.float32)
  return tf.nn.softmax(decedent_mtx)

def get_decedent_mtx_from_score(heads):
  return tf.nn.softmax(tf.transpose(heads, perm=[0, 2, 1]))

def one_hot(heads):
  return tf.one_hot(heads, tf.shape(heads)[-1], on_value=constants.VERY_LARGE,
             off_value=constants.VERY_SMALL)
def local_window_balanced(input, strip_width):
  strip_width = int(strip_width)
  # tf.roll()
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-strip_width, strip_width+1)], axis=0)
  diag = tf.where(tf.greater(diag, 0), constants.VERY_LARGE * tf.ones_like(diag, dtype=tf.float32),
                  constants.VERY_SMALL * tf.ones_like(diag, dtype=tf.float32))
  return tf.cast(diag, tf.float32)

def local_window_rtilted(input, strip_width):
  strip_width = int(strip_width)
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-(strip_width-1), strip_width+1)], axis=0)
  diag = tf.where(tf.greater(diag, 0), constants.VERY_LARGE * tf.ones_like(diag, dtype=tf.float32),
                  constants.VERY_SMALL * tf.ones_like(diag, dtype=tf.float32))
  return tf.cast(diag, tf.float32)

def local_window_ltilted(input, strip_width):
  strip_width = int(strip_width)
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-strip_width, strip_width)], axis=0)
  diag = tf.where(tf.greater(diag, 0), constants.VERY_LARGE * tf.ones_like(diag, dtype=tf.float32), constants.VERY_SMALL * tf.ones_like(diag, dtype=tf.float32))
  return tf.cast(diag, tf.float32)

def gen_block_by_line(idx, len, size, offset):
  array_idx = tf.range(len, dtype=tf.int32)
  array_location =tf.math.logical_and(tf.greater_equal(array_idx, idx-offset), tf.less(array_idx, idx+size-offset))
  # line = tf.where(array_location, tf.ones_like(array_idx, dtype=tf.float32),
  #                 tf.zeros_like(array_idx, dtype=tf.float32))
  line = tf.where(array_location, constants.VERY_LARGE * tf.ones_like(array_idx, dtype=tf.float32), constants.VERY_SMALL * tf.ones_like(array_idx, dtype=tf.float32))
  # line = tf.Print(line, [line], "line")
  return line
def gen_block_by_instance(input, idxer):
  seq_len = input.get_shape()[0]
  size = tf.cast(input / 12, dtype=tf.int32)
  offset = tf.cast(input % 12, dtype=tf.int32)
  mtx = tf.map_fn(lambda inp: gen_block_by_line(idx = inp[0], len = seq_len, size = inp[1], offset = inp[2]), (idxer, size, offset), dtype=tf.float32)
  return tf.cast(mtx, tf.float32)
def chunk_to_block_diag(input):
  seq_len = input.get_shape()[1]
  idxer = tf.range(seq_len, dtype=tf.int32)
  batch = tf.map_fn(lambda inp: gen_block_by_instance(inp, idxer), elems=input, dtype=tf.float32)
  return tf.cast(batch, tf.float32)




dispatcher = {
  'one_hot': one_hot,
  'local_window_balanced': local_window_balanced,
  'local_window_ltilted': local_window_ltilted,
  'local_window_rtilted': local_window_rtilted,
  'chunk_to_block_diag': chunk_to_block_diag,
  'get_decedent_mtx': get_decedent_mtx,
  'get_decedent_mtx_from_score': get_decedent_mtx_from_score
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined transformation function %s' % fn_name)
    exit(1)

def get_params(input, transformation_name, src_name):
  transformation_diag_width_in_name = ['local_window_balanced',
                                 'local_window_ltilted',
                                 'local_window_rtilted']
  transformation_pass_through = ['one_hot', 'get_decedent_mtx']
  if transformation_name in transformation_pass_through:
    return {'heads': input}
  elif transformation_name in transformation_diag_width_in_name:
    return {'input': input, 'strip_width': src_name.split('_')[-1]}
  else:
    print('Undefined transformation param format')
    raise NotImplementedError

