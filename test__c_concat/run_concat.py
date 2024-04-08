import paddle
import numpy as np

ranks = 7
dtype = 'bfloat16'
path = '/workspace/PaddleApiTest/test__c_concat/data/inputs_{index}.npy'

data_list = [np.load(path.format(index=i)) for i in range(ranks)]
data_list = [paddle.to_tensor(data) for data in data_list]
if dtype == 'bfloat16':
    data_list = [paddle.cast(data, 'uint16') for data in data_list]
concated_data = paddle.concat(data_list, axis=1)
if dtype == 'bfloat16':
    concated_data = paddle.cast(concated_data, 'float32')
concated_data = concated_data.numpy()
np.save('/workspace/PaddleApiTest/test__c_concat/data/concat.npy', concated_data)



