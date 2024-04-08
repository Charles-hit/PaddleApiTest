import numpy as np


concat_data = np.load('/workspace/PaddleApiTest/test__c_concat/data/concat.npy')
_c_concat_data = np.load('/workspace/PaddleApiTest/test__c_concat/data/_c_concat.npy')
print(concat_data.shape)
print(_c_concat_data.shape)
print(concat_data.dtype)
print(_c_concat_data.dtype)
np.testing.assert_equal(concat_data, _c_concat_data)