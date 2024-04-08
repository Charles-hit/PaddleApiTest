import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.layers.mpu.mp_ops import _c_concat
import numpy as np

ranks = 7
dtype = 'bfloat16'
path = '/workspace/PaddleApiTest/test__c_concat/data/inputs_{index}.npy'

dist.init_parallel_env()

data = np.load(path.format(index=dist.get_rank()))
data = paddle.to_tensor(data)
if dtype == 'bfloat16':
    data = paddle.cast(data, 'uint16')
concated_data = _c_concat(data)
if dtype == 'bfloat16':
    concated_data = paddle.cast(concated_data, 'float32')
concated_data = concated_data.numpy()
np.save('/workspace/PaddleApiTest/test__c_concat/data/_c_concat.npy', concated_data)

