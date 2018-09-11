import tensorflow as tf


def get_meshgrid(images):
    batch, height, width, _ = tf.unstack(tf.shape(images))
    gb, gy, gx = tf.meshgrid(tf.range(batch), tf.range(height), tf.range(width),
                             indexing = 'ij')
    return gb, gy, gx


def bilinear_warp(images, flow):
    _, h, w, _ = tf.unstack(tf.shape(images))
    gb, gy, gx = get_meshgrid(images)
    gb = tf.cast(gb, tf.float32)
    gy = tf.cast(gy, tf.float32)
    gx = tf.cast(gx, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # Calculate warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(gy+fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(gy+fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(gx+fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(gx+fx_1, 0., w_lim)

    g_00 = tf.cast(tf.stack([gb, gy_0, gx_0], axis = -1), tf.int32)
    g_01 = tf.cast(tf.stack([gb, gy_0, gx_1], axis = -1), tf.int32)
    g_10 = tf.cast(tf.stack([gb, gy_1, gx_0], axis = -1), tf.int32)
    g_11 = tf.cast(tf.stack([gb, gy_1, gx_1], axis = -1), tf.int32)

    # Warp pixels
    x_00 = tf.gather_nd(images, g_00)
    x_01 = tf.gather_nd(images, g_01)
    x_10 = tf.gather_nd(images, g_10)
    x_11 = tf.gather_nd(images, g_11)

    # Calculate interpolation weights
    w_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = -1)
    w_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = -1)
    w_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = -1)
    w_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = -1)

    return w_00*x_00 + w_01*x_01 + w_10*x_10 + w_11*x_11


class DeepVoxelFlow(object):
    def __init__(self, name = 'dvf'):
        self.name = name

    def __call__(self, images_0, images_1, t):
        with tf.variable_scope(self.name) as vs:
            images = tf.concat([images_0, images_1], axis = -1)
            conv1 = tf.layers.Conv2D(64, (5, 5), (1, 1), 'same')(images)
            conv1 = tf.layers.BatchNormalization()(conv1)
            conv1 = tf.nn.relu(conv1)

            conv2 = tf.layers.MaxPooling2D((2, 2), (2, 2), 'same')(conv1)
            conv2 = tf.layers.Conv2D(128, (5, 5), (1, 1), 'same')(conv2)
            conv2 = tf.layers.BatchNormalization()(conv2)
            conv2 = tf.nn.relu(conv2)

            conv3 = tf.layers.MaxPooling2D((2, 2), (2, 2), 'same')(conv2)
            conv3 = tf.layers.Conv2D(256, (3, 3), (1, 1), 'same')(conv3)
            conv3 = tf.layers.BatchNormalization()(conv3)
            conv3 = tf.nn.relu(conv3)

            conv4 = tf.layers.MaxPooling2D((2, 2), (2, 2), 'same')(conv3)
            conv4 = tf.layers.Conv2D(256, (3, 3), (1, 1), 'same')(conv4)
            conv4 = tf.layers.BatchNormalization()(conv4)
            conv4 = tf.nn.relu(conv4)

            _, h, w, _ = tf.unstack(tf.shape(conv4))
            conv4_up = tf.image.resize_bilinear(conv4, (2*h, 2*w))
            conv5 = tf.concat([conv4_up, conv3], axis = -1)
            conv5 = tf.layers.Conv2D(256, (3, 3), (1, 1), 'same')(conv5)
            conv5 = tf.layers.BatchNormalization()(conv5)
            conv5 = tf.nn.relu(conv5)

            _, h, w, _ = tf.unstack(tf.shape(conv5))
            conv5_up = tf.image.resize_bilinear(conv5, (2*h, 2*w))
            conv6 = tf.concat([conv5_up, conv2], axis = -1)
            conv6 = tf.layers.Conv2D(128, (5, 5), (1, 1), 'same')(conv6)
            conv6 = tf.layers.BatchNormalization()(conv6)
            conv6 = tf.nn.relu(conv6)

            _, h, w, _ = tf.unstack(tf.shape(conv6))
            conv6_up = tf.image.resize_bilinear(conv6, (2*h, 2*w))
            conv7 = tf.concat([conv6_up, conv1], axis = -1)
            conv7 = tf.layers.Conv2D(64, (5, 5), (1, 1), 'same')(conv7)
            conv7 = tf.layers.BatchNormalization()(conv7)
            conv7 = tf.nn.relu(conv7)

            outputs = tf.layers.Conv2D(3, (5, 5), (1, 1), 'same')(conv7)
            outputs = tf.nn.tanh(outputs)

            flow = outputs[:, :, :, :2]
            mask = tf.expand_dims(outputs[:, :, :, 2], axis = 3)

            # Rescale the estimated optical flow to the actual size
            _, h, w, _ = tf.unstack(tf.shape(images))
            fx, fy = tf.unstack(flow, axis = -1)
            fx *= tf.cast(w, tf.float32)/2
            fy *= tf.cast(h, tf.float32)/2
            flow = tf.stack([fx, fy], axis = -1)

            # Scale flow along time-axis
            t = tf.reshape(t, (-1, 1, 1, 1))
            flow_tto0 = (-1)*t*flow
            flow_tto1 = (1-t)*flow

            # Warp pixel both from former and from latter frames
            images_tfrom0 = bilinear_warp(images_0, flow_tto0)
            images_tfrom1 = bilinear_warp(images_1, flow_tto1)
            
            # Frame synthesis
            mask = 0.5*(1.0 + mask)
            images_t = mask*images_tfrom0 + (1.0 - mask)*images_tfrom1

            return images_t, flow

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
