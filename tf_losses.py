import tensorflow as tf

def policy_loss(target, output):
    # Illegal moves are marked by a value of -1 in the labels - we mask these with large negative values
    output = tf.where(target < 0, -1e5, output)
    # The large negative values will still break the loss, so we replace them with 0 once we finish masking
    target = tf.nn.relu(target)
    # The stop gradient is maybe paranoia, but it can't hurt
    policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target), logits=output)
    return tf.reduce_mean(input_tensor=policy_cross_entropy)


def value_loss(target, output):
    output = tf.cast(output, tf.float32)
    value_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(target), logits=output)
    return tf.reduce_mean(input_tensor=value_cross_entropy)


def moves_left_loss(target, output):
    # Scale the loss to similar range as other losses.
    scale = 20.0
    target = target / scale
    output = tf.cast(output, tf.float32) / scale
    huber = tf.keras.losses.Huber(10.0 / scale)
    return tf.reduce_mean(huber(target, output))
