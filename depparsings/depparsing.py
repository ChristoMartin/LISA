#%%
import sys
from supar import Parser
#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#%%
section = 'dev'
#%%

with open('conll05.{}.plain'.format(section)) as f:
    lines = f.readlines()
    sents = [line.strip().split(' ') for line in lines]
#%%

with open('conll05.{}.heads'.format(section)) as f:
	lines = f.readlines()
	sents_heads = [line.strip().split(' ') for line in lines]
	sents_heads = [[int(head) for head in sent] for sent in sents_heads]
with open('conll05.{}.biaffine-gold.heads'.format(section)) as f:
	lines = f.readlines()
	mix_sents_heads = [line.strip().split(' ') for line in lines]
	mix_sents_heads = [[int(head) for head in sent] for sent in mix_sents_heads]
#%%

def one_hot(head):
	# print(head)
	b = np.zeros((head.size+1, head.size+1))
	b[np.arange(head.size)+1, head] = 1
	# print(b.shape)
	return b

#%%|
parser_biaffine = Parser.load('biaffine-dep-en')
parser_biaffine_bert = Parser.load('biaffine-dep-bert-en')
parser_crf_dep_en = Parser.load('crf-dep-en')
parser_crf2o_dep_en = Parser.load('crf2o-dep-en')
parser_crfnp_dep_en = Parser.load('crfnp-dep-en')

#%%
sents = sents
sents = ['it proved a perfect time for Radio Free Europe to ask for permission to set up office .']
sents = [i.split(' ') for i in sents]
# sents = [['The', 'seeds', 'already', 'are', 'in', 'the', 'script', '.']]
sents_heads = [[2, 0, 6, 6, 2, 12, 10, 10, 12, 12, 2, 12, 13, 16, 14, 16, 16, 2]]

#%%
dataset_biaffine = parser_biaffine.predict(sents, prob=True, verbose=False)
# dataset_biaffine_bert = parser_biaffine_bert.predict(sents, prob=True, verbose=False)
dataset_crf2o_dep_en = parser_crf2o_dep_en.predict(sents, prob=True, verbose=False)
dataset_crf_dep_en = parser_crf_dep_en.predict(sents, prob=True, verbose=False)
print(len(sents))
dataset_crfnp_dep_en = parser_crfnp_dep_en.predict(sents, prob=True, verbose=False)
#%%
attn_biaffine = [one_hot(np.array(arc)) for arc in dataset_biaffine.arcs]
# attn_biaffine_bert = [one_hot(np.array(arc)) for arc in dataset_biaffine_bert.arcs]
# attn_crf2o_dep_en = [one_hot(np.array(arc)) for arc in dataset_crf2o_dep_en.arcs]
attn_crf_dep_en = [one_hot(np.array(arc)) for arc in dataset_crf_dep_en.arcs]
attn_crfnp_dep_en = [one_hot(np.array(arc)) for arc in dataset_crfnp_dep_en.arcs]
attn_gold = [one_hot(np.array(arc)) for arc in sents_heads]
# attn_biaffine_gold = [one_hot(np.array(arc)) for arc in mix_sents_heads]
#%%
# a = np.random.random((16, 16))
# plt.imshow(a, cmap='hot', interpolation='nearest')
# plt.show()

#%%
# dataset = [(np.array(biaf)+np.array(crf2o)+np.array(crfnp))/3 for biaf, crf2o, crfnp in zip(dataset_biaffine.arcs, dataset_crf2o_dep_en.arcs, dataset_crfnp_dep_en.arcs)]

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

#%%
aggregated_heatmap_loss = [0., 0.]
majority_heatmap_loss = [0., 0.]


for heatmap, sent in zip(attn_biaffine, sents):
	sns.heatmap(heatmap, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent)
	plt.show()

for heatmap, sent in zip(attn_crfnp_dep_en, sents):
	sns.heatmap(heatmap, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent)
	plt.show()




for heatmap, sent in zip(attn_crf_dep_en, sents):
	# f, axes = plt.subplots(1, 2, figsize=(30, 20))
	# axes[0].set_title('Aggregated graph')
	# axes[1].set_title('Gold graph')
		# print(head)
	sns.heatmap(heatmap, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent)
	# sns.heatmap(head, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent, ax=axes[1])
	# plt.savefig("integrated_heatmap/{}_sent{}.jpeg".format(section, i+1))
	# plt.close(f)
	plt.show()

for hm1, hm2, hm3, sent in zip(attn_biaffine, attn_crfnp_dep_en, attn_crf_dep_en, sents):
	sns.heatmap((hm1+hm2+hm3)/3, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent)
	plt.show()



# for arc_biaf, arc_crf2o, arc_crf, sent, i, head in zip(attn_biaffine, attn_crf2o_dep_en, attn_crf_dep_en, sents, range(len(sents)), attn_gold):
# 	# print(arc_biaf_bert)
# 	# print(np.array(arc_biaf).shape, np.array(arc_biaf_bert).shape, np.array(arc_crf2o).shape, np.array(arc_crfnp).shape)
# 	# print(one_hot(np.array(arc_biaf)).shape, one_hot(np.array(arc_crf2o)).shape, one_hot(np.array(arc_crfnp)).shape)
# 	# heatmap = (one_hot(np.array(arc_biaf)) + one_hot(np.array(arc_biaf_bert)) +one_hot(np.array(arc_crf2o))+one_hot(np.array(arc_crfnp)))/4
# 	heatmap = (arc_biaf + arc_crf2o + arc_crf)/3
# 	# print(heatmap)
# 	# print(np.amax(heatmap, axis=1))
# 	majority = one_hot(np.argmax((arc_biaf + arc_crf2o + arc_crf), axis=1)[1:])
# 	# print(heatmap)
# 	if np.random.rand() > 1:
# 		f, axes = plt.subplots(1, 2, figsize=(30, 20))
# 		axes[0].set_title('Aggregated graph')
# 		axes[1].set_title('Gold graph')
# 		# print(head)
# 		sns.heatmap(heatmap, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent, ax=axes[0])
# 		sns.heatmap(head, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent, ax=axes[1])
# 		plt.savefig("integrated_heatmap/{}_sent{}.jpeg".format(section, i+1))
# 		plt.close(f)
# 	# ax.plot()
# 	# plt.show()
# 	aggregated_heatmap_loss[0] += np.sum(np.linalg.norm(heatmap-head, axis=-1))#
# 	aggregated_heatmap_loss[1] += cross_entropy(heatmap, head)#np.sum(np.linalg.norm(heatmap-head, axis=-1))
# 	majority_heatmap_loss[0] += np.sum(np.linalg.norm(majority-head, axis=-1))
# 	majority_heatmap_loss[1] += cross_entropy(majority, head)
	# break

#%%
# for name, ind_parser in (("biaffine", attn_biaffine),
# 				   # ("biaffine_bert", attn_biaffine_bert),
# 				   #("crf_2o", attn_crf2o_dep_en),
# 				   #("crf", attn_crf_dep_en)):
# 						 ("biaffine-gold mix", attn_biaffine_gold),
# 					("crfnp", attn_crfnp_dep_en)):
# 	heatmap_loss = [0. for _ in range(2)]
# 	for arc, sent, i, head in zip(ind_parser, sents, range(len(sents)), attn_gold):
# 		heatmap = arc#(arc_biaf + arc_biaf_bert + arc_crf2o + arc_crfnp)/4
# 		if np.random.rand()>1:
# 			f, axes = plt.subplots(1, 2, figsize=(30, 20))
# 			axes[0].set_title('biaffine graph')
# 			axes[1].set_title('Gold graph')
# 			sns.heatmap(heatmap, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent, ax=axes[0])
# 			sns.heatmap(head, linewidth=0.5, xticklabels=['root']+sent, yticklabels=['root']+sent, ax=axes[1])
#
# 			plt.savefig("{}_heatmap/{}_sent{}.jpeg".format(name, section, i+1))
# 			plt.close(f)
# 		# print(heatmap, head)
# 		heatmap_loss[0] += np.average(np.linalg.norm(heatmap-head, axis=-1))
# 		heatmap_loss[1] += cross_entropy(heatmap, head)#np.sum(np.linalg.norm(heatmap-head, axis=-1))
# 		# break
# 	print("{} 2Norm loss: {}".format(name, heatmap_loss[0]/len(sents_heads)))
# 	print("{} CE loss: {}".format(name, heatmap_loss[1] / len(sents_heads)))

#%%
# print("biaf loss: {}".format(biaf_heatmap_loss/len(sents_heads)))
print("aggregated 2Norm loss: {}".format(aggregated_heatmap_loss[0]/len(sents_heads)))
print("aggregated CE loss: {}".format(aggregated_heatmap_loss[1]/len(sents_heads)))
print("majority 2Norm loss: {}".format(majority_heatmap_loss[0]/len(sents_heads)))
print("majority CE loss: {}".format(majority_heatmap_loss[1]/len(sents_heads)))
#%%


