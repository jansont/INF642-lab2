
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

num_batches = len(dataset_train)
val_batches = len(dataset_validate)


def generator(dataset):
    while True:
        for F0, bert, AU1, AU2, AU4 in dataset:
             yield ([F0, AU1[:, :-1], AU2[:, :-1], AU4[:,:-1]], [AU1[:,1:], AU2[:,1:],AU4[:,1:]])


 
hist = Model.fit(x = generator(dataset_train), validation_data = generator(dataset_validate), epochs=200, steps_per_epoch = num_batches, validation_steps =val_batches).history
