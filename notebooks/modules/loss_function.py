import tensorflow as tf

#######################################################################################
#    LOSS FUNCTION                                                                    #
#######################################################################################
def compute_loss(x, x_out, mean, log_var):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_out)
    cross_entropy = -tf.math.reduce_sum(cross_entropy, axis=[1,2,3])

    regularization = .5 * tf.math.reduce_sum(1 + log_var - tf.math.square(mean) - tf.math.exp(log_var), axis=1)
    loss = tf.math.reduce_mean(cross_entropy + regularization)
    return -loss