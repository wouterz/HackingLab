import tensorflow as tf
import numpy as np

labels = [1,2,3,4,5]

def label_mapping():
    imagenet_label = np.zeros([3, 3])
    imagenet_label[0:3, 0:3] = np.eye(3)
    return tf.constant(imagenet_label, dtype=tf.float32)


def loss_tensor(y_true, y_pred):
    # print('true', y_true)
    # print('pred', y_pred)
    # print('map', label_mapping())

    disLogits = tf.matmul(y_pred, label_mapping())
    # print('disLogits', disLogits)

    softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=disLogits)
    # print('softmax', softmax)
    cross_entropy_loss = tf.reduce_mean(
        softmax
        )
    # reg_loss = 2e-6 * tf.nn.l2_loss(self.model.get_layer('adv_layer_1').adv_weights)
    # loss = cross_entropy_loss + reg_loss
    loss = cross_entropy_loss
    return loss

y_true = label_mapping()
print('true', y_true)
test_loss = loss_tensor(y_true, tf.constant([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]], dtype=tf.float32 ) )
print('testloss_0', test_loss)

test_loss = loss_tensor(y_true, tf.constant([[1,0,0,0,0],[0,1.0,0,0,0],[0,0,0,1,0]], dtype=tf.float32 ) )
print('testloss_ouside', test_loss)

test_loss = loss_tensor(y_true, tf.constant([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]], dtype=tf.float32 ) )
print('testloss_same class', test_loss)

test_loss = loss_tensor(y_true, tf.constant([[1,0,0,0,0],[0,0.5,0.5,0,0],[0,0.5,0.5,0,0]], dtype=tf.float32 ) )
print('testloss_same_split', test_loss)

test_loss = loss_tensor(y_true, tf.constant([[1,0,0,0,0],[0,1,0,0,0],[0,0,0.5,0,0.5]], dtype=tf.float32 ) )
print('testloss_split_out_1', test_loss)

test_loss = loss_tensor(y_true, tf.constant([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,00.5,0.5]], dtype=tf.float32 ) )
print('testloss_split_out_2', test_loss)

# test_loss = loss_tensor(tf.constant([[1,0,0],[0,1,0],[0,0,1]], dtype=tf.float32), tf.constant([[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0]], dtype=tf.float32 ) )
# print('testloss_double', test_loss)

# test_loss = loss_tensor(tf.constant([[1,0,0],[0,1,0],[0,0,1]], dtype=tf.float32), tf.constant([[1,0,0,0,0],[0,1,0,0,0],[0,0.5,0.5,0,0]], dtype=tf.float32 ) )
# print('testloss_split', test_loss)

# test_loss = loss_tensor(tf.constant([[1,0,0],[0,1,0],[0,0,1]], dtype=tf.float32), tf.constant([[1,0,0],[1,0,0],[1,0,0]], dtype=tf.float32 ) )
# print('testloss_same', test_loss)