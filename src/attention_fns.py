import tensorflow as tf
import constants
import nn_utils


def copy_from_predicted(mode, train_attention_to_copy, eval_attention_to_copy):
  attention_to_copy = train_attention_to_copy if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_to_copy

  # check whether this thing is actually scores or if it's predictions, and needs
  # to be expanded out to one-hot scores. If it's actually scores, dims should be
  # batch x batch_seq_len x batch_seq_len, and thus rank should be 3
  if len(attention_to_copy.get_shape()) < 3:
    # use non-standard on and off values because we're going to softmax this later, and want the result to be 0/1
    attention_to_copy = tf.one_hot(attention_to_copy, tf.shape(attention_to_copy)[-1], on_value=constants.VERY_LARGE,
                                   off_value=constants.VERY_SMALL)

  return tf.cast(attention_to_copy, tf.float32)

def linear_aggregation(mode, train_attention_aggregation, eval_attention_aggregation):
  #suppose attention_to_aggregated is in list
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated = tf.map_fn(lambda src: tf.one_hot(src, tf.shape(src)[-1], on_value=constants.VERY_LARGE,
                                 off_value=constants.VERY_SMALL), elems=attention_to_aggregated, dtype=tf.float32)
  attention_to_aggregated = nn_utils.graph_aggregation_softmax_done(attention_to_aggregated)
  return tf.cast(attention_to_aggregated, tf.float32)
def mean_aggregation(mode, train_attention_aggregation, eval_attention_aggregation):
  #suppose attention_to_aggregated is in list
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated = tf.map_fn(lambda src: tf.one_hot(src, tf.shape(src)[-1], on_value=constants.VERY_LARGE,
                                 off_value=constants.VERY_SMALL), elems=attention_to_aggregated, dtype=tf.float32)
  attention_to_aggregated = nn_utils.graph_mean_aggregation(attention_to_aggregated)
  return tf.cast(attention_to_aggregated, tf.float32)


dispatcher = {
  'copy_from_predicted': copy_from_predicted,
  'linear_aggregation': linear_aggregation,
  'mean_aggregation': mean_aggregation
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined attention function `%s' % fn_name)
    exit(1)


def get_params(mode, attn_map, train_outputs, features, labels):
  params = {'mode': mode}
  params_map = attn_map['params']
  # print("debug <attention fn get parameter>: ", params, params_map, features, labels)
  for param_name, param_values in params_map.items():
    # if this is a map-type param, do map lookups and pass those through
    if param_name == "train_attention_aggregation":
      params[param_name] = tf.stack([labels[src] for src in param_values['label']], axis=0)
    elif param_name == "eval_attention_aggregation":
      params[param_name] = tf.stack([labels[src] for src in param_values['label']], axis=0)
    elif 'label' in param_values:
      params[param_name] = labels[param_values['label']]
    elif 'feature' in param_values:
      params[param_name] = features[param_values['feature']]
    # otherwise, this is a previous-prediction-type param, look those up and pass through
    elif 'layer' in param_values:
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]
    else:
      params[param_name] = param_values['value']
  # print("debug <attention fn parameters>: ", params)
  return params
