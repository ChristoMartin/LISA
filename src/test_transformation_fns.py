import tensorflow as tf
import numpy as np
import output_fns
from src import attention_fns
import transformation_fn

class OutputFnTests(tf.test.TestCase):

	# correct solution:
	def softmax(self, x):
		"""Compute softmax values for each set of scores in x."""
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum(axis=0)


	def test_transformation(self):
		# raise NotImplemented
		with self.test_session():
			# code = size *12 + offset
			block_list = tf.constant([[24, 25, 36, 37, 38, 0], [24, 25, 24, 25, 0, 0]], dtype=tf.int32)
			line = transformation_fn.chunk_to_block_diag(block_list)
			print(line.eval())
			# weight = tf.constant([0.2, 0.3, 0.5])
			# dependency_list_weight_pair = (dependency_list, weight)
			# attention = attention_fns.attention_to_aggregated(mode=tf.estimator.ModeKeys.TRAIN, train_attention_to_aggregated=dependency_list_weight_pair, eval_attention_to_aggregated=None)
			# print(attention.eval())


if __name__ == '__main__':

	tf.test.main()
