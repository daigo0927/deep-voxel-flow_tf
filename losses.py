import tensorflow as tf

def L1loss(images_gt, images_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(images_gt-images_pred), axis = -1))

