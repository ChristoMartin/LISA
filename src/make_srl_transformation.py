import sys
from collections import Counter, OrderedDict


class Argument:
  def __init__(self, tag, start_idx, end_idx):
    self.tag = tag
    self.start_idx = start_idx
    self.end_idx = end_idx
    self.is_gold = False

  def __str__(self):
    return "({}: {} - {})".format(self.tag, self.start_idx, self.end_idx)

  def __repr__(self):
    return self.__str__()

  def _check_equality_boundary(self, rarg):
    if self.start_idx == rarg.start_idx and self.end_idx == rarg.end_idx:
      return True
    else:
      return False

  def _check_overlap(self, rarg):
    if (self.end_idx<rarg.start_idx) or\
      (self.start_idx>rarg.end_idx):
      return True
    else:
      return False


    # if (self.end_idx >= rarg.start_idx and self.start_idx <= rarg.end_idx and ) or \
    #   (self.start_idx <= rarg.end_idx and self.end_idx >= rarg.start_idx) or \
    #   (self.start_idx <= rarg.start_idx and self.end_idx >= rarg.end_idx) or \
    #   (self.start_idx >= rarg.start_idx and self.end_idx <= rarg.end_idx):
    #   return True
    # else:
    #   return False


# %%

def read_srl_05(filename):
  with open(filename) as f:
    lines = f.readlines()
  lines = [line.split() for line in lines]
  container = []
  sents = []
  for line in lines:
    if len(line) > 0:
      container.append(line)
    else:
      sents.append(container)
      container = []
  data = []
  for sent in sents:
    slots = [[] for _ in range(len(sent[0]))]
    for token in sent:
      for idx in range(len(sent[0])):
        slots[idx].append(token[idx])
    data.append(slots)
  sents = data
  pred_argument_maps = []
  for sent in sents:

    predicate = list(filter(lambda x: x != '-', sent[0]))
    predicate_str = sent[0]
    pred_argument_map = OrderedDict({})
    # pred: []
    pred_cnt = 0
    for pred in predicate:
      pred_argument_map["{}_{}".format(pred, pred_cnt)] = []
      pred_cnt+=1
    pred_argument_map["PRED##STR"] = predicate_str
    arguments = sent[1:]
    length = len(sent[0])
    for pred_idx in range(len(predicate)):
      container = []
      for tok, id in zip(arguments[pred_idx], range(length)):
        if len(container) == 0 and tok == '*':
          continue
        elif len(container) > 0 and tok == '*':
          for item in container:
            item[2] += 1
        elif tok.startswith('(') and tok.endswith('*'):
          container.append([tok[1:-1], id, 0])
        elif tok.startswith('(') and tok.endswith('*)'):
          pred_argument_map["{}_{}".format(predicate[pred_idx], pred_idx)].append(Argument(tok[1:-2], id, id))
        # print(Argument(tok[1:-2], id, id+1))
        elif tok == '*)':
          item_to_add = container.pop()
          pred_argument_map["{}_{}".format(predicate[pred_idx], pred_idx)].append(
            Argument(item_to_add[0], item_to_add[1], item_to_add[2] + item_to_add[1]))
    pred_argument_maps.append(pred_argument_map)
  return pred_argument_maps


gdata = read_srl_05(sys.argv[2])

pdata = read_srl_05(sys.argv[1])


def arguments2str_conversion(args, len):
  # args = sorted(args, key = lambda x: x[0])
  converted_str = ['*' for _ in range(len)]
  for arg in args:
    if arg.start_idx == arg.end_idx:
      # print(arg.tag, arg.start_idx, arg.end_idx)
      converted_str[arg.start_idx] = "({}*)".format(arg.tag)
    else:
      converted_str[arg.start_idx] = "({}*".format(arg.tag)
      converted_str[arg.end_idx] = "*)"
      # converted_str[arg.start_idx + 1:arg.end_idx] = ["*" for _ in range(arg.end_idx - arg.start_idx + 1 - 2)]
  return converted_str


# %%

def back_conversion(data):
  print_strs = []
  for item in data:
    pred_str = item["PRED##STR"]
    sent_len = len(pred_str)
    # print(sent_len)
    container = [pred_str]
    pstr = []
    for pred, args in item.items():
      if pred == "PRED##STR":
        continue
      container.append(arguments2str_conversion(args, sent_len))
    for idx in range(sent_len):
      pstr.append('\t'.join([item[idx] for item in container]))
    print_strs.append(pstr)
    # print(pstr)
  return print_strs


# %%

def fix_labels(pred_args, gold_args):
  for parg in pred_args:
    for garg in gold_args:
      if parg._check_equality_boundary(garg):
        parg.tag = garg.tag
  return pred_args


# %%


core_args_tag = set(['A{}'.format(idx) for idx in range(6)])#.union(set(['C-A{}'.format(idx) for idx in range(6)]))
# print(core_args_tag)

def move_arg(pred_args, gold_args):
  # get unique arguments
  def get_unique_core_arguments_as_gold(args, gold_args):
    args_tag = map(lambda x: x.tag, args)
    tag_args_map = {arg.tag: arg for arg in args}
    gold_tags = [arg.tag for arg in gold_args]
    # print(gold_tags)
    unique_args = []
    others = []
    args_count = Counter(args_tag)
    for arg_tag, count in args_count.items():
      # print(arg_tag,  arg_tag in core_args_tag)
      if count == 1 and arg_tag in core_args_tag and arg_tag in gold_tags:
        unique_args.append(tag_args_map[arg_tag])
      else:
        others.append(tag_args_map[arg_tag])
    return unique_args, others


  # print(pred_args)
  # print(gold_args)
  uniqc_gold_args, _ = get_unique_core_arguments_as_gold(gold_args, gold_args)
  uniqc_pred_args, p_others = get_unique_core_arguments_as_gold(pred_args, uniqc_gold_args)
  tag_uniqc_gold_arg_map = {arg.tag: arg for arg in uniqc_gold_args}
  # print("<uniq core argument>: ", uniqc_pred_args)
  # print("<gold uniq core argument>: ", uniqc_gold_args)
  # print()
  for arg in uniqc_pred_args:
    # if arg.tag in tag_uniqc_gold_arg_map.keys():
    arg.start_idx = tag_uniqc_gold_arg_map[arg.tag].start_idx
    arg.end_idx = tag_uniqc_gold_arg_map[arg.tag].end_idx
    for other_arg in p_others:
      if arg.start_idx <= other_arg.start_idx and arg.end_idx >= other_arg.end_idx:
        pred_args.remove(other_arg)
      elif arg.start_idx > other_arg.start_idx and arg.end_idx < other_arg.end_idx:
        pred_args.append(Argument(other_arg.tag, arg.end_idx + 1, other_arg.end_idx))
        other_arg.end_idx = arg.start_idx - 1
      elif arg.start_idx <= other_arg.start_idx and arg.end_idx < other_arg.end_idx and arg.end_idx>=other_arg.start_idx:
        other_arg.start_idx = arg.end_idx + 1
      elif arg.start_idx > other_arg.start_idx and arg.end_idx >= other_arg.end_idx and arg.start_idx<=other_arg.end_idx:
        other_arg.end_idx = arg.start_idx - 1
      else:
        continue


  return pred_args


# %%

def merge_spans(pred_args, gold_args):
  p_end_idx_arg_map = {arg.end_idx: arg for arg in pred_args}
  p_start_idx_arg_map = {arg.start_idx: arg for arg in pred_args}
  g_tag_arg_map = {arg.tag: arg for arg in gold_args}
  for eidx, arg in p_end_idx_arg_map.items():
    if eidx + 2 in p_start_idx_arg_map.keys():
      pls = p_start_idx_arg_map[eidx + 2]
      if pls.tag == arg.tag and arg.tag in g_tag_arg_map.keys():
        if (arg.start_idx == g_tag_arg_map[arg.tag].start_idx \
          and pls.end_idx == g_tag_arg_map[arg.tag].end_idx):
          pred_args.append(Argument(arg.tag, arg.start_idx, g_tag_arg_map[arg.tag].end_idx))
          pred_args.remove(arg)
          pred_args.remove(p_start_idx_arg_map[eidx + 2])
  return pred_args


# %%

def split_spans(pred_args, gold_args):
  g_end_idx_arg_map = {arg.end_idx: arg for arg in gold_args}
  g_start_idx_arg_map = {arg.start_idx: arg for arg in gold_args}
  p_tag_arg_map = {arg.tag: arg for arg in pred_args}
  for eidx, arg in g_end_idx_arg_map.items():
    if eidx + 2 in g_start_idx_arg_map.keys():
      gls = g_start_idx_arg_map[eidx + 2]
      if gls.tag == arg.tag and arg.tag in p_tag_arg_map.keys():
        if arg.start_idx == p_tag_arg_map[arg.tag].start_idx \
          and gls.end_idx == p_tag_arg_map[arg.tag].end_idx:
          pred_args.append(Argument(arg.tag, arg.start_idx, arg.end_idx))
          pred_args.append(Argument(gls.tag, gls.start_idx, gls.end_idx))
          pred_args.remove(p_tag_arg_map[arg.tag])
  return pred_args


# %%

def fix_boundary(pred_args, gold_args):
  for parg in pred_args:
    for garg in gold_args:
      # print(parg, garg)
      if parg.tag == garg.tag and parg._check_overlap(garg):
        parg.start_idx = garg.start_idx
        parg.end_idx = garg.end_idx
        for other_arg in pred_args:
          if other_arg == parg:
            continue
          if parg.start_idx<=other_arg.start_idx and parg.end_idx>=other_arg.end_idx:
            pred_args.remove(other_arg)
          elif parg.start_idx>other_arg.start_idx and parg.end_idx<other_arg.end_idx:
            pred_args.append(Argument(other_arg.tag, parg.end_idx+1,other_arg.end_idx))
            other_arg.end_idx = parg.start_idx-1
          elif parg.start_idx<=other_arg.start_idx and parg.end_idx<other_arg.end_idx and parg.end_idx>=other_arg.start_idx:
            other_arg.start_idx = parg.end_idx+1
          elif parg.start_idx>other_arg.start_idx and parg.end_idx>=other_arg.end_idx and parg.start_idx<=other_arg.end_idx:
            other_arg.end_idx = parg.start_idx-1
          else:
            continue

  return pred_args


def drop_arg(pred_args, gold_args):
  drop_ind = [(any([parg._check_overlap(garg) for garg in gold_args]), parg) for parg in pred_args]
  for ind, parg in drop_ind:
    if not ind:
      pred_args.remove(parg)
  return pred_args


def add_arg(pred_args, gold_args):
  add_ind = [(any([garg._check_overlap(parg) for parg in pred_args]), garg) for garg in gold_args]
  for ind, garg in add_ind:
    if not ind:
      pred_args.append(garg)
  return pred_args


# %%
def write_back(bc_str, filename):
  with open(filename, "w") as f:
    for sent in bc_str:
      for tok in sent:
        f.write(tok)
        f.write('\n')
      f.write('\n')




# %%
transformation_list = ['fix_labels', 'move_arg', 'merge_spans', 'split_spans', 'fix_boundary', 'drop_arg', 'add_arg']

transformations = OrderedDict({
  'fix_labels': fix_labels,
  'move_arg': move_arg,
  'merge_spans': merge_spans,
  'split_spans': split_spans,
  'fix_boundary': fix_boundary,
  'drop_arg': drop_arg,
  'add_arg': add_arg
})

# %% Extracting arguments


for t_name in transformation_list:
# for t_name, t_func in transformations.items():
  t_func = transformations[t_name]
  for predicted, gold in zip(pdata, gdata):
    # predicated/gold: a dict containing all (predicate:arguments) paris
    for pred, args in predicted.items():
      if pred == "PRED##STR":
        continue
      if not pred in gold.keys():
        continue
      gold_args = gold[pred]
      # print("performing {} transformation on {}".format(t_name, pred))
      # print("gold arguments:", gold_args)
      # print("arguments before transformation {}: ".format(t_name), args)
      args = t_func(args, gold_args)
      # print("{}'s arguments after transformation {}: ".format(pred, t_name), args)

  p_bc_str = back_conversion(pdata)
  g_bc_str = back_conversion(gdata)
  write_back(p_bc_str, '{}.t.{}'.format(sys.argv[1], t_name))
  write_back(g_bc_str, '{}.t.{}'.format(sys.argv[2], t_name))
