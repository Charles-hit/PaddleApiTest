import pickle
import logging
import numpy as np

class Pickle(object):
  @staticmethod
  def load(path):
    with open(path, 'rb') as file:
      return pickle.load(file)

  @staticmethod
  def save(obj, path):
    with open(path, 'wb') as file:
      pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

class Result(object):
  def __init__(self, mode, data):
    self.mode = mode
    self.data = data

  def assert_equal(self, other):
    if self.mode == other.mode:
      logging.info("-- assert_equal=self:{}, other:{}".format(self.data, other.data))
      return np.testing.assert_equal(self.data, other.data)
    else:
      logging.info("-- assert_allclose=self:{}, other:{}".format(self.data, other.data))
      return np.testing.assert_allclose(self.data, other.data, rtol=1e-6)
