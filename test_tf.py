import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())

with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Run on the GPU
    c = tf.matmul(a, b)
    print(c)
