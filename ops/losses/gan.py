import tensorflow as tf


def generator_loss(d_fake,
                   metrics='JSD'):
    if metrics in ['JSD', 'jsd']:
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake),
                                                    logits=d_fake))
    elif metrics in ['WD', 'wd']:
        return -tf.reduce_mean(d_fake)
    elif metrics in ['HINGE', 'hinge']:
        return -tf.reduce_mean(d_fake)
    elif metrics in ['LS', 'ls', 'PearsonChiSquared', 'PCS', 'pcs']:
        return tf.reduce_mean((d_fake-1)**2)/2
    else:
        raise ValueError


def discriminator_loss(d_real,
                       d_fake,
                       metrics='JSD'):
    if metrics in ['JSD', 'jsd']:
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real),
                                                    logits=d_real))
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake),
                                                    logits=d_fake))
        return real_loss + fake_loss
    elif metrics in ['WD', 'wd']:
        return -(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))
    elif metrics in ['HINGE', 'hinge']:
        real_loss = -tf.reduce_mean(tf.minimum(0, -1 + d_real))
        fake_loss = -tf.reduce_mean(tf.minimum(0, -1 - d_fake))
        return real_loss + fake_loss
    elif metrics in ['LS', 'ls', 'PearsonChiSquared', 'PCS', 'pcs']:
        return tf.reduce_mean((d_real-1)**2)/2 + tf.reduce_mean((d_fake-0)**2)/2
    else:
        raise ValueError


def gradient_penalty(discriminator,
                     real,
                     fake):
    bs = real.get_shape().as_list()[0]
    if len(real.get_shape().as_list()) == 4:
        eps = tf.random_uniform(shape=[bs, 1, 1, 1],
                                minval=0., maxval=1.)
        reduction_indices = [1, 2, 3]
    elif len(real.get_shape().as_list()) == 2:
        eps = tf.random_uniform(shape=[bs, 1],
                                minval=0., maxval=1.)
        reduction_indices = [1]
    else:
        raise ValueError

    differences = fake - real
    interpolates = real + (eps * differences)

    with tf.GradientTape() as g:
        g.watch(interpolates)
        y = discriminator(interpolates,
                          training=True)
    grads = g.gradient(y, interpolates)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads),
                                   reduction_indices=reduction_indices))
    gp = tf.reduce_mean(tf.square(slopes - 1.))
    return gp


def discriminator_norm(d_real):
    with tf.name_scope('DiscriminatorNorm'):
        return tf.nn.l2_loss(d_real)
