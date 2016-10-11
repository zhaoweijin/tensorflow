# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test for tensorflow.python.ops.variable_scope."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables

class VarDefaultInitializerTest(tf.test.TestCase):

	def testZerosInitializer(self):

		TYPES = [tf.int8, tf.uint8, tf.int16, tf.uint16, tf.int32, tf.int64, 
						tf.bool, tf.complex64, tf.complex128]
		i = 0
		for dtype in TYPES:
			x = tf.get_variable(name='x'+str(i), shape=(3,4), dtype=dtype)
			y = tf.get_variable(name='y'+str(i), dtype=dtype, 
							initializer=init_ops.zeros_initializer(shape=(3,4), dtype=dtype))
			
			with self.test_session() as sess:
				sess.run(tf.initialize_all_variables())
				self.assertAllEqual(x.eval(), y.eval())
			
			i = i + 1


	def testStringsInitializer(self):

		dtype = tf.string

		x = tf.get_variable(name='x', shape=(3,4), dtype=dtype)
		y = tf.get_variable(name='y', dtype=dtype, 
			initializer=init_ops.strings_initializer(shape=(3,4), dtype=dtype))
		
		with self.test_session() as sess:
			sess.run(tf.initialize_all_variables())
			self.assertAllEqual(x.eval(), y.eval())


class PartitionedVarDefaultInitializerTest(tf.test.TestCase):

	def testZerosInitializer(self):

		TYPES = [tf.int8, tf.uint8, tf.int16, tf.uint16, tf.int32, tf.int64, 
						tf.bool, tf.complex64, tf.complex128]
		i = 0
		for dtype in TYPES:
			x = tf.get_variable(name='xx'+str(i), shape=(3,4), dtype=dtype,
							partitioner=partitioned_variables.min_max_variable_partitioner())
			y = tf.get_variable(name='yy'+str(i), dtype=dtype, 
							partitioner=partitioned_variables.min_max_variable_partitioner(),
							initializer=init_ops.zeros_initializer(shape=(3,4), dtype=dtype))
			
			with self.test_session() as sess:
				sess.run(tf.initialize_all_variables())
				val_x = sess.run(x._get_variable_list())
				val_y = sess.run(y._get_variable_list())

				self.assertAllEqual(val_x, val_y)
			
			i = i + 1

	def testStringsInitializer(self):

		dtype = tf.string

		x = tf.get_variable(name='xx', shape=(3,4), dtype=dtype,
							partitioner=partitioned_variables.min_max_variable_partitioner())
		y = tf.get_variable(name='yy', dtype=dtype, 
			partitioner=partitioned_variables.min_max_variable_partitioner(),
			initializer=init_ops.strings_initializer(shape=(3,4), dtype=dtype))
		
		with self.test_session() as sess:
			sess.run(tf.initialize_all_variables())
			val_x = sess.run(x._get_variable_list())
			val_y = sess.run(y._get_variable_list())

			self.assertAllEqual(val_x, val_y)


if __name__ == "__main__":
  tf.test.main()
