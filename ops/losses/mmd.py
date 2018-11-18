import tensorflow as tf


def mmd(x,
        y,
        kernel='rbf',
        **kernel_params):
    """

    Args:
        x (tf.Tensor): shape (bs, N)
        y (tf.Tensor): shape (bs, N)
        kernel (str): kernel function
        bias (bool):  biased or unbiased
        **kernel_params: parameters of kernel

    Returns:
        mmd_loss

    """
    bs = x.get_shape().as_list()[0]
    half_bs = bs*(bs-1)//2
    norm_x = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    dot_xx = tf.matmul(x, x, transpose_b=True)
    dis_xx = norm_x + tf.transpose(norm_x) - 2*dot_xx

    norm_y = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
    dot_yy = tf.matmul(y, y, transpose_b=True)
    dis_yy = norm_y + tf.transpose(norm_y) - 2*dot_yy

    dot_xy = tf.matmul(x, y, transpose_b=True)
    dis_xy = norm_x + tf.transpose(norm_y) - 2*dot_xy

    if kernel in ['gaussian', 'rbf', 'RBF']:
        sigma2_k = tf.nn.top_k(
            tf.reshape(dis_xy, [-1]), half_bs).values[half_bs - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(dis_xx, [-1]), half_bs).values[half_bs - 1]

        res1 = tf.exp(- dis_xx / 2. / sigma2_k)
        res1 += tf.exp(- dis_yy / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(bs))
        res1 = tf.reduce_sum(res1) / (bs * (bs - 1))
        res2 = tf.exp(- dis_xy / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (bs * bs)
        stat = res1 - res2
    elif kernel in ['IMQ']:
        raise NotImplementedError
    else:
        raise ValueError
    return stat
