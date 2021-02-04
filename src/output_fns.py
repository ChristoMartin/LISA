from collections import OrderedDict

import tensorflow as tf
from tensorflow.estimator import ModeKeys
import nn_utils
import tf_utils


def softmax_classifier(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):

  with tf.name_scope('softmax_classifier'):

    # # todo pass this as initial proj dim (which is optional)
    # projection_dim = model_config['predicate_pred_mlp_size']
    #
    # with tf.variable_scope('MLP'):
    #   mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
    with tf.variable_scope('Classifier'):
      logits = nn_utils.MLP(inputs, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

    # todo implement this
    if transition_params is not None:
      print('Transition params not yet supported in softmax_classifier')
      exit(1)

    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    targets_onehot = tf.one_hot(indices=targets, depth=num_labels, axis=-1)
    loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(logits, [-1, num_labels]),
                                           onehot_labels=tf.reshape(targets_onehot, [-1, num_labels]),
                                           weights=tf.reshape(tokens_to_keep, [-1]),
                                           label_smoothing=hparams.label_smoothing,
                                           reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    # loss = tf.Print(loss, [loss], ' softmax classifier')
    #
    # loss = 0.
    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    # loss = tf.zeros([])
    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': logits,
      'probabilities': tf.nn.softmax(logits, -1)
    }

  return output


def get_separate_scores_preds_from_joint(joint_outputs, joint_maps, joint_num_labels):
  predictions = joint_outputs['predictions']
  scores = joint_outputs['scores']
  output_shape = tf.shape(predictions)
  batch_size = output_shape[0]
  batch_seq_len = output_shape[1]
  sep_outputs = {}
  for map_name, label_comp_map in joint_maps.items():
    # print("debug <joint map>: name {}, comp_map {}".format(map_name, label_comp_map))
    short_map_name = map_name.split('_to_')[-1]
    # label_comp_predictions = tf.nn.embedding_lookup(label_comp_map, predictions)
    # sep_outputs["%s_predictions" % short_map_name] = tf.squeeze(label_comp_predictions, -1)

    # marginalize out probabilities for this task
    task_num_labels = tf.shape(tf.unique(tf.reshape(label_comp_map, [-1]))[0])[0]
    joint_probabilities = tf.nn.softmax(scores)
    joint_probabilities_flat = tf.reshape(joint_probabilities, [-1, joint_num_labels])
    # print("debug <computing segment_ids>! ")
    # tf.Print("debug <@get_separate_scores_preds_from_joint>:", tf.nn.embedding_lookup(label_comp_map, tf.range(joint_num_labels)))
    segment_ids = tf.squeeze(tf.nn.embedding_lookup(label_comp_map, tf.range(joint_num_labels)), -1)
    segment_scores = tf.unsorted_segment_sum(tf.transpose(joint_probabilities_flat), segment_ids, task_num_labels)
    segment_scores = tf.reshape(tf.transpose(segment_scores), [batch_size, batch_seq_len, task_num_labels])
    sep_outputs["%s_probabilities" % short_map_name] = segment_scores

    # use marginalized probabilities to get predictions
    sep_outputs["%s_predictions" % short_map_name] = tf.argmax(segment_scores, -1)
  # print("debug <sep output>: ", sep_outputs)
  return sep_outputs


def joint_softmax_classifier(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, joint_maps,
                             transition_params):

  with tf.name_scope('joint_softmax_classifier'):

    # todo pass this as initial proj dim (which is optional)
    projection_dim = model_config['predicate_pred_mlp_size']

    with tf.variable_scope('MLP'):
      mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
    with tf.variable_scope('Classifier'):
      logits = nn_utils.MLP(mlp, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

    # todo implement this
    if transition_params is not None:
      print('Transition params not yet supported in joint_softmax_classifier')
      exit(1)

    # print("debug <shape of joint softmax classifier logit vs. target>", logits, " ", targets)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

    # print("debug <shape of joint softmax classifier ce vs. tokens_to_keep>", cross_entropy, " ", tokens_to_keep)
    cross_entropy *= tokens_to_keep
    loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)
    # loss = 0.
    # loss = tf.Print(loss, [loss], ' joint softmax classifier')

    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': logits,
      'probabilities': tf.nn.softmax(logits, -1)
    }

    # now get separate-task scores and predictions for each of the maps we've passed through
    separate_output = get_separate_scores_preds_from_joint(output, joint_maps, num_labels)
    combined_output = OrderedDict({**output, **separate_output})



    return combined_output


def parse_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  class_mlp_size = model_config['class_mlp_size']
  attn_mlp_size = model_config['attn_mlp_size']

  if transition_params is not None:
    print('Transition params not supported in parse_bilinear')
    exit(1)

  with tf.variable_scope('parse_bilinear'):
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2,
                                       keep_prob=hparams.mlp_dropout)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

    with tf.variable_scope('Arcs'):
      # batch_size x batch_seq_len x batch_seq_len
      arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)

    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(arc_logits, -1)
    probabilities = tf.nn.softmax(arc_logits)

    # arc_logits = tf.Print(arc_logits, [arc_logits], 'parse_bilinear_arc_logit')
    # targets = tf.Print(targets, [])
    # targets = tf.Print(targets, [targets, tf.reduce_sum(targets)], 'parse_bilinear_targets, checking whether the target vector is passed correctly')

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'parse_ce_loss')
    loss = tf.reduce_sum((cross_entropy * tokens_to_keep))/ num_tokens
    # loss = 0.
    # loss = tf.zeros([])
    # loss = tf.Print(loss, [loss], 'parse_bilinear')
    output = {
      'loss': loss,
      'predictions': predictions,
      'probabilities': probabilities,
      'scores': arc_logits,
      'dep_rel_mlp': dep_rel_mlp,
      'head_rel_mlp': head_rel_mlp
    }

  return output


def conditional_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params,
                         dep_rel_mlp, head_rel_mlp, parse_preds_train, parse_preds_eval):

  parse_preds = parse_preds_train if mode == ModeKeys.TRAIN else parse_preds_eval
  with tf.variable_scope('conditional_bilin'):
    logits, _ = nn_utils.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_labels,
                                                         parse_preds, hparams.bilinear_dropout)

  predictions = tf.argmax(logits, -1)
  probabilities = tf.nn.softmax(logits)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

  n_tokens = tf.reduce_sum(tokens_to_keep)
  loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / n_tokens
  # loss = tf.zeros([])
  output = {
    'loss': loss,
    'scores': logits,
    'predictions': predictions,
    'probabilities': probabilities
  }

  return output


def srl_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, predicate_preds_train,
                 predicate_preds_eval, predicate_targets, transition_params):
    '''

    :param input: Tensor with dims: [batch_size, batch_seq_len, hidden_size]
    :param predicate_preds: Tensor of predictions from predicates layer with dims: [batch_size, batch_seq_len]
    :param targets: Tensor of SRL labels with dims: [batch_size, batch_seq_len, batch_num_predicates]
    :param tokens_to_keep:
    :param predictions:
    :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
    :return:
    '''
    # inputs = tf.Print(inputs, [tf.shape(inputs)], "bilinear input size")
    with tf.name_scope('srl_bilinear'):

      def bool_mask_where_predicates(predicates_tensor):
        return tf.logical_and(tf.not_equal(predicates_tensor, predicate_outside_idx), tf.cast(tokens_to_keep, tf.bool))

      input_shape = tf.shape(inputs)
      batch_size = input_shape[0]
      batch_seq_len = input_shape[1]

      predicate_mlp_size = model_config['predicate_mlp_size']
      role_mlp_size = model_config['role_mlp_size']

      # TODO this should really be passed in, not assumed...
      predicate_outside_idx = 0

      predicate_preds = predicate_preds_train if mode == tf.estimator.ModeKeys.TRAIN else predicate_preds_eval
      # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "srl-bilinear predicate prediction")
      predicate_gather_indices = tf.where(bool_mask_where_predicates(predicate_preds))



      # (1) project into predicate, role representations
      with tf.variable_scope('MLP'):
        predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size, keep_prob=hparams.mlp_dropout)
        predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                  predicate_role_mlp[:, :, predicate_mlp_size:]

      # predicate_mlp = tf.Print(predicate_mlp, [predicate_mlp, tf.shape(predicate_mlp)],
                                       # "srl-bilinear predicate_mlp shape")
      # role_mlp = tf.Print(role_mlp, [role_mlp, tf.shape(role_mlp)],
      #                          "srl-bilinear role_mlp shape")
      # predicate_gather_indices = tf.Print(predicate_gather_indices, [predicate_gather_indices], "srl-bilinear gather-indices")

      # (2) feed through bilinear to obtain scores
      with tf.variable_scope('Bilinear'):
        # gather just the predicates
        # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
        # role mlp: batch x seq_len x role_mlp_size
        # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
        # i.e. a (num_predicates_in_batch x batch_seq_len x role_mlp_size) tensor
        gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
        tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                                 [batch_size, batch_seq_len, batch_seq_len, role_mlp_size])
        gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)
        # gathered_predicates = tf.Print(gathered_predicates, [tf.shape(gathered_predicates)], "srl-bilinear gathered_predicates shape")
        # gathered_roles = tf.Print(gathered_roles, [tf.shape(gathered_roles)],
        #                                "srl-bilinear gathered_roles shape")
        # now multiply them together to get (num_predicates_in_batch x batch_seq_len x num_srl_classes) tensor of scores
        srl_logits = nn_utils.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels,
                                                       hparams.bilinear_dropout)
        # srl_logits = tf.Print(srl_logits, [srl_logits, tf.shape(srl_logits)], "srl-bilinear logit")
        srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])

      # (3) compute loss

      # need to repeat each of these once for each target in the sentence
      mask_tiled = tf.reshape(tf.tile(tokens_to_keep, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
      mask = tf.gather_nd(mask_tiled, predicate_gather_indices)

      # now we have k sets of targets for the k frames
      # (p1) f1 f2 f3
      # (p2) f1 f2 f3

      # get all the tags for each token (which is the predicate for a frame), structuring
      # targets as follows (assuming p1 and p2 are predicates for f1 and f3, respectively):
      # (p1) f1 f1 f1
      # (p2) f3 f3 f3
      srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

      gold_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_targets), tf.int32), -1)
      srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

      # num_predicates_in_batch x seq_len
      srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

      predicted_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_preds), tf.int32), -1)
      srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
      srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

      # num_predicates_in_batch x seq_len
      predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

      seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

      if transition_params is not None and (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL):
        predictions, score = tf.contrib.crf.crf_decode(srl_logits_transposed, transition_params, seq_lens)

      if transition_params is not None and mode == ModeKeys.TRAIN and tf_utils.is_trainable(transition_params):
        # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(srl_logits_transposed,
                                                                              srl_targets_predicted_predicates,
                                                                              seq_lens, transition_params)
        loss = tf.reduce_mean(-log_likelihood)
      else:
        srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
        loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                               onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                               weights=tf.reshape(mask, [-1]),
                                               label_smoothing=hparams.label_smoothing,
                                               reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      # predictions = tf.Print(predictions, [predictions, tf.shape(predictions)], "srl-bilinear: prediction")
      # srl_logits_transposed = tf.Print(srl_logits_transposed, [srl_logits_transposed, tf.shape(srl_logits_transposed)], "srl-bilinear logit")

      output = {
        'loss': loss,
        'predictions': predictions,
        'scores': srl_logits_transposed,
        'targets': srl_targets_gold_predicates,
        'probabilities': tf.nn.softmax(srl_logits_transposed, -1)
      }

      return output


dispatcher = {
  'srl_bilinear': srl_bilinear,
  'joint_softmax_classifier': joint_softmax_classifier,
  'softmax_classifier': softmax_classifier,
  'parse_bilinear': parse_bilinear,
  'conditional_bilinear': conditional_bilinear,
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined output function `%s' % fn_name)
    exit(1)


# need to decide shape/form of train_outputs!
def get_params(mode, model_config, task_map, train_outputs, features, labels, current_outputs, task_labels, num_labels,
               joint_lookup_maps, tokens_to_keep, transition_params, hparams):
  params = {'mode': mode, 'model_config': model_config, 'inputs': current_outputs, 'targets': task_labels,
            'tokens_to_keep': tokens_to_keep, 'num_labels': num_labels, 'transition_params': transition_params,
            'hparams': hparams}
  params_map = task_map['params'] if 'params' in task_map else {}
  for param_name, param_values in params_map.items():
    # print(param_name, ": ", param_values)

    # if this is a map-type param, do map lookups and pass those through
    if 'joint_maps' in param_values:
      params[param_name] = {map_name: joint_lookup_maps[map_name] for map_name in param_values['joint_maps']}
    elif 'label' in param_values:
      params[param_name] = labels[param_values['label']]
    elif 'feature' in param_values:
      params[param_name] = features[param_values['feature']]
    # otherwise, this is a previous-prediction-type param, look those up and pass through
    elif 'layer' in param_values:
      # print("debug <train outputs>: ", train_outputs)
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]
    else:
      params[param_name] = param_values['value']
    # print("debug <get params, returned params>:", params)
  return params
