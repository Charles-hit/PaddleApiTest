import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.layers.mpu.mp_ops import _c_identity
from paddle.distributed import collective
import numpy as np

ranks = 7
dtype = 'float32'
path = '/workspace/PaddleApiTest/test__c_identity/data/inputs_{index}.npy'

dist.init_parallel_env()

np_data = np.load(path.format(index=dist.get_rank()))
data = paddle.to_tensor(np_data)
if dtype == 'bfloat16':
    data = paddle.cast(data, 'uint16')
group = collective._get_default_group()
identity_data = _c_identity(data, group)
np_identity_data = identity_data.numpy()
print(data.numpy())
print(np_identity_data)
np.testing.assert_equal(np_identity_data, data.numpy())

