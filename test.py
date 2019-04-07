import tensorflow as tf
import numpy as np
tensor = tf.Variable([[2,2,2], [2,3,4],[1,2,3]], dtype = tf.float32)
t_mean = tf.Variable([2,5], dtype = tf.float32)
c = tf.reduce_mean(t_mean,axis = 0)
print (t_mean.get_shape)
w = np.asarray([1,2,3])
print (type(w))
o = np.var(w)
print (o)
'''
l = list()
for i in range(3):
  l.append(tensor)
q = tf.add_n(l)
'''
a,b = tf.nn.moments(tensor,axes = [1])
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print(sess.run(c))

# Output
