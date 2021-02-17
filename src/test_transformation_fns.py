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
			dependency_list = tf.constant([[1, 24, 4, 4, 1, -1], [1, 24, 4, 4, 1, 3]])
			b1 = transformation_fn.local_window_balanced(dependency_list, 1)
			b2 = transformation_fn.local_window_balanced(dependency_list, 2)
			l1 = transformation_fn.local_window_ltilted(dependency_list, 1)
			l2 = transformation_fn.local_window_ltilted(dependency_list, 2)
			r1 = transformation_fn.local_window_rtilted(dependency_list, 1)
			r2 = transformation_fn.local_window_rtilted(dependency_list, 2)
			print(b1.eval())
			print(b2.eval())
			print(l1.eval())
			print(l2.eval())
			print(r1.eval())
			print(r2.eval())
			# weight = tf.constant([0.2, 0.3, 0.5])
			# dependency_list_weight_pair = (dependency_list, weight)
			# attention = attention_fns.attention_to_aggregated(mode=tf.estimator.ModeKeys.TRAIN, train_attention_to_aggregated=dependency_list_weight_pair, eval_attention_to_aggregated=None)
			# print(attention.eval())


if __name__ == '__main__':

	tf.test.main()
