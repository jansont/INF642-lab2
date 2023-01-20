import tensorflow as tf


def encode(E1, E2, E3, E4, E5, E6, E7):
    return E1, E2, E3, E4, E5, E6, E7


def tf_encode(x, z1, z2,z4, z5, z6, z7):
    result_x, result_z1, result_z2, result_z4, result_z5, result_z6, result_z7 = tf.py_function(encode, [x, z1, z2,z4, z5, z6, z7], [tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,tf.int64,tf.int64])

    result_x.set_shape([None])
    result_z1.set_shape([None])
    result_z2.set_shape([None])
    result_z4.set_shape([None])
    result_z5.set_shape([None])
    result_z6.set_shape([None])
    result_z7.set_shape([None])

    return result_x, result_z1, result_z2, result_z4, result_z5, result_z6, result_z7

BUFFER_SIZE = 20000
BATCH_SIZE = 64

# Creating dataset
dataset_train = tf.data.Dataset.from_tensor_slices((F0train, AU1_train, AU2_train, AU4_train, AU5_train, AU6_train, AU7_train))
dataset_validate = tf.data.Dataset.from_tensor_slices((F0validate, AU1_validate, AU2_validate, AU4_validate, AU5_validate, AU6_validate, AU7_validate))
dataset_test = tf.data.Dataset.from_tensor_slices((F0test, AU1_test, AU2_test, AU4_test, AU5_test, AU6_test, AU7_test))
dataset_train = dataset_train.map(tf_encode)
dataset_train = dataset_train.cache()
dataset_train = dataset_train.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validate = dataset_validate.map(tf_encode)
dataset_validate = dataset_validate.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
dataset_test = dataset_test.map(tf_encode)
dataset_test = dataset_test.cache()
dataset_test = dataset_test.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE_TEST)
