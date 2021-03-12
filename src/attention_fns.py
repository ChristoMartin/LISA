import tensorflow as tf
import constants
import nn_utils
import transformation_fn


def copy_from_predicted(mode, train_attention_to_copy, eval_attention_to_copy):
  attention_to_copy = train_attention_to_copy if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_to_copy

  # check whether this thing is actually scores or if it's predictions, and needs
  # to be expanded out to one-hot scores. If it's actually scores, dims should be
  # batch x batch_seq_len x batch_seq_len, and thus rank should be 3
  if len(attention_to_copy.get_shape()) < 3:
    # use non-standard on and off values because we're going to softmax this later, and want the result to be 0/1
    attention_to_copy = tf.one_hot(attention_to_copy, tf.shape(attention_to_copy)[-1], on_value=constants.VERY_LARGE,
                                   off_value=constants.VERY_SMALL)

  return tf.cast(tf.nn.softmax(attention_to_copy, dim=-1), tf.float32), None

def linear_aggregation(mode, train_attention_aggregation, eval_attention_aggregation, parser_dropout=0.9):
  #suppose attention_to_aggregated is in list
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated, weight = nn_utils.graph_aggregation_softmax_done(attention_to_aggregated, parser_dropout)
  return tf.cast(attention_to_aggregated, tf.float32), weight
def mean_aggregation(mode, train_attention_aggregation, eval_attention_aggregation, parser_dropout=0.9):
  #suppose attention_to_aggregated is in list
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated = nn_utils.graph_mean_aggregation(attention_to_aggregated, parser_dropout)
  return tf.cast(attention_to_aggregated, tf.float32), None

def linear_aggregation_by_mlp(mode, train_attention_aggregation, eval_attention_aggregation, v, mlp_dropout, projection_dim, parser_dropout=0.9, batch_norm=False):
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  aggregated_attention, weight = nn_utils.graph_mlp_aggregation(attention_to_aggregated, v, mlp_dropout, projection_dim, parser_dropout, batch_norm)

  # raise NotImplementedError
  return tf.cast(aggregated_attention, tf.float32), weight


dispatcher = {
  'copy_from_predicted': copy_from_predicted,
  'linear_aggregation': linear_aggregation,
  'mean_aggregation': mean_aggregation,
  'linear_aggregation_mlp': linear_aggregation_by_mlp
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined attention function `%s' % fn_name)
    exit(1)


def get_params(mode, attn_map, train_outputs, features, labels, hparams, model_config):
  params = {'mode': mode}
  params_map = attn_map['params']
  # if attn_map['name']
  # print("debug <attention fn get parameter>: ", params, params_map, features, labels)
  for param_name, param_values in params_map.items():
    # if this is a map-type param, do map lookups and pass those through
    if 'label' in param_values:
      if isinstance(param_values['label'], dict):
        attn_constraints = []
        for src, transformation_name in param_values['label'].items():
          attn_map = transformation_fn.dispatch(transformation_name)(**transformation_fn.get_params(labels[src], transformation_name, src))
          attn_constraints += [attn_map]
        params[param_name] = tf.stack(attn_constraints, axis=1)
        # params['num_dep_graphs']
      elif isinstance(param_values['label'], list): # only for compatability reason
        params[param_name] = tf.stack([labels[src] for src in param_values['label']], axis=1)
      elif isinstance(param_values['label'], str): # only for compatability reason
        params[param_name] = labels[param_values['label']]
      else:
        print('Undefined attention source format')
        raise NotImplementedError
      # todo sentence feature may be invoked by non-aggregation attentions
      params['parser_dropout'] = hparams.parser_dropout
      if hparams.aggregator_mlp_bn:
        params['batch_norm'] = True
      if 'sentence_feature' in param_values:
        params['mlp_dropout'] = hparams['mlp_dropout']
        params['projection_dim'] = model_config['linear_aggregation_scorer_mlp_size']
        params['v'] = features['sentence_feature']
    elif 'output' in param_values:
      if isinstance(param_values['output'], dict):
        outputs = []
        for layer_name, output_name in param_values['output'].items():
          outputs_layer = train_outputs[layer_name]
          if isinstance(output_name, list):
            #Here, uses (output_name, transformation_fn_name) pair
            output = transformation_fn.dispatch(output_name[1])(outputs_layer[output_name[0]])
            outputs += [output]
          else:
            outputs += [outputs_layer[output_name]]
        params[param_name] = tf.stack(outputs, axis=1)
      else:
        raise NotImplementedError
      params['parser_dropout'] = hparams.parser_dropout
      if hparams.aggregator_mlp_bn:
        params['batch_norm'] = True
    elif 'feature' in param_values:
      if isinstance(param_values['feature'], dict):
        attn_constraints = []
        for src, transformation_name in param_values['feature'].items():
          attn_map = transformation_fn.dispatch(transformation_name)(
            **transformation_fn.get_params(features[src], transformation_name, src))
          attn_constraints += [attn_map]
        params[param_name] = tf.stack(attn_constraints, axis=1)
      elif isinstance(param_values['feature'], list):  # only for compatability reason
        params[param_name] = tf.stack([features[src] for src in param_values['feature']], axis=1)
      elif isinstance(param_values['feature'], str):  # only for compatability reason
        params[param_name] = features[param_values['label']]
      else:
        print('Undefined attention source format')
        raise NotImplementedError
      # todo sentence feature may be invoked by non-aggregation attentions
      params['parser_dropout'] = hparams.parser_dropout
      if hparams.aggregator_mlp_bn:
        params['batch_norm'] = True
      if 'sentence_feature' in param_values:
        params['mlp_dropout'] = hparams.mlp_dropout
        params['projection_dim'] = model_config['linear_aggregation_scorer_mlp_size']
        params['v'] = features['sentence_feature']
    elif 'layer' in param_values:
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]
    else:
      params[param_name] = param_values['value']
  # print("debug <attention fn parameters>: ", params)
  return params
