import tensorflow as tf


# 其中 mask 是一个索引向量，值为1表示该位置的标签在训练数据中是给定的；比如100个数据中训练集已知带标签的数据有50个，
# 那么计算损失的时候，loss 乘以的 mask  等于 loss 在未带标签的地方都乘以0没有了，而在带标签的地方损失变成了mask倍；
# 即只对带标签的样本计算损失。
# 注：loss的shape与mask的shape相同，等于样本的数量：(None,），所以 loss *= mask 是向量点乘。


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
    # return tf.reduce_mean(accuracy_all), tf.argmax(preds, 1)


