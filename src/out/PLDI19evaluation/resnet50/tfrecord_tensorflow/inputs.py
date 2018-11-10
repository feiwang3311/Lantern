import tensorflow as tf
from tensorflow import data


class Pipeline(object):
  def __init__(self, args, sess):
    self.sess = sess
    self.batch_size = args.batch_size

    # target_image_size = (args.target_image_size if hasattr(args, 'target_image_size') else None)

    training_dataset = self._create_dataset(
      batch_size=args.batch_size,
      pad_batch=False,
      repeat=None,
      num_input_threads=1,
      shuffle=False,
      shuffle_buffer=None,
      # seed=None,
      files=args.train_tfrecord_filepaths,
      # distort_image=False,
      target_image_size=(args.batch_size, 3, 32, 32)
    )

    # validation_dataset = self._create_dataset(
    #   batch_size=args.batch_size * args.num_gpus,
    #   pad_batch=True,
    #   repeat=1,
    #   num_input_threads=args.num_input_threads,
    #   shuffle=False,
    #   shuffle_buffer=None,
    #   files=args.validation_tfrecord_filepaths,
    #   distort_image=False,
    #   target_image_size=target_image_size
    # )

    self._handle = tf.placeholder(tf.string, shape=[])
    self._is_training = tf.placeholder(tf.bool, [], 'is_training')
    self.iterator = data.Iterator.from_string_handle(
      self._handle, training_dataset.output_types, training_dataset.output_shapes
    )

    self.data = self.iterator.get_next()

    self.training_iterator = training_dataset.make_initializable_iterator()
    self.sess.run(self.training_iterator.initializer)
    # self.validation_iterator = validation_dataset.make_initializable_iterator()
    # self.initialize_validation_data()

    self._training_handle = sess.run(self.training_iterator.string_handle())
    # self._validation_handle = sess.run(self.validation_iterator.string_handle())

  @property
  def is_training(self):
    return self._is_training

  @property
  def training_data(self):
    return {self._handle: self._training_handle, self._is_training: True}

  # @property
  # def validation_data(self):
  #   return {self._handle: self._validation_handle,
  #       self._is_training: False}

  # def initialize_validation_data(self):
  #   self.sess.run(self.validation_iterator.initializer)

  @staticmethod
  def _create_dataset(batch_size,
            pad_batch,
            repeat,
            num_input_threads,
            shuffle,
            shuffle_buffer,
            files,
            seed=1337,
            distort_image=None,
            target_image_size=None):
    assert batch_size % 2 == 0

    input_processor = _InputProcessor(
      batch_size=batch_size,
      num_threads=num_input_threads,
      repeat=repeat,
      shuffle=shuffle,
      shuffle_buffer=shuffle_buffer,
      seed=seed,
      distort_image=distort_image,
      target_image_size=target_image_size
    )

    dataset = input_processor.from_tfrecords(files)
    if pad_batch:
      dataset = dataset.padded_batch(
        batch_size=1,
        padded_shapes=_get_padded_shapes(dataset.output_shapes, batch_size),
        padding_values=_get_padded_types(dataset.output_types)
      ).apply(tf.contrib.data.unbatch())
    return dataset


def _get_padded_shapes(output_shapes, batch_size):
  feature_shapes = dict()
  for feature, shape in output_shapes[0].items():
    feature_dims = shape.dims[1:]
    feature_shapes[feature] = tf.TensorShape(
      [tf.Dimension(batch_size)] + feature_dims)
  return feature_shapes, batch_size


def _get_padded_types(output_types):
  feature_values = dict()
  for feature, feature_type in output_types[0].items():
    feature_values[feature] = tf.constant(-1, feature_type)
  return feature_values, tf.constant(-1, tf.int64)


class _InputProcessor(object):
  def __init__(self,
         batch_size,
         num_threads,
         repeat,
         shuffle,
         shuffle_buffer,
         seed,
         distort_image=None,
         target_image_size=None):
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.repeat = repeat
    self.shuffle = shuffle
    self.shuffle_buffer = shuffle_buffer
    self.seed = seed
    self.distort_image = distort_image
    self.target_image_size = target_image_size

  def from_tfrecords(self, files):
    dataset = data.TFRecordDataset(files)
    dataset = dataset.map(
      map_func=self._preprocess_example,
      num_parallel_calls=self.num_threads
    )
    dataset = dataset.repeat(self.repeat)
    if self.shuffle:
      dataset = dataset.shuffle(
        buffer_size=self.shuffle_buffer,
        seed=self.seed
      )
    dataset = dataset.batch(self.batch_size)
    return dataset

  def _preprocess_example(self, serialized_example):
    parsed_example = self._parse_serialized_example(serialized_example)
    image = self._preprocess_image(parsed_example['image'])
    return {'image': image}, parsed_example['label']

  def _preprocess_image(self, raw_image):
    image = tf.decode_raw(raw_image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [3, 32, 32])
    # image = tf.image.decode_jpeg(raw_image, channels=3)
    #image = tf.image.resize_images(image, tf.constant([32, 32])) # self.target_image_size)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    # if self.distort_image:
    #   image = tf.image.random_flip_left_right(image)
    # image = tf.transpose(image, [2, 0, 1])
    return image

  @staticmethod
  def _parse_serialized_example(serialized_example):
    features = {
      'image': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    }
    return tf.parse_single_example(serialized=serialized_example, features=features)
