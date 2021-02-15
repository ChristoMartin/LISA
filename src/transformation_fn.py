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



def one_hot(heads):
  return tf.one_hot(heads, tf.shape(heads)[-1], on_value=constants.VERY_LARGE,
             off_value=constants.VERY_SMALL)
def local_window_balanced(input, strip_width):
  strip_width = int(strip_width)
  # tf.roll()
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-strip_width, strip_width+1)], axis=0)
  return tf.cast(diag, tf.float32)

def local_window_rtilted(input, strip_width):
  strip_width = int(strip_width)
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-(strip_width-1), strip_width+1)], axis=0)
  return tf.cast(diag, tf.float32)

def local_window_ltilted(input, strip_width):
  strip_width = int(strip_width)
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-strip_width, strip_width)], axis=0)
  return tf.cast(diag, tf.float32)



dispatcher = {
  'one_hot': one_hot,
  'local_window_balanced': local_window_balanced,
  'local_window_ltilted': local_window_ltilted,
  'local_window_rtilted': local_window_rtilted
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined transformation function `%s' % fn_name)
    exit(1)

def get_params(input, transformation_name, src_name):
  transformation_diag_width_in_name = ['local_window_balanced',
                                 'local_window_ltilted',
                                 'local_window_rtilted']
  transformation_pass_through = ['one_hot']
  if transformation_name in transformation_pass_through:
    return {'heads': input}
  elif transformation_name in transformation_diag_width_in_name:
    return {'input': input, 'strip_width': src_name.split('_')[-1]}
  else:
    print('Undefined transformation param format')
    raise NotImplementedError

