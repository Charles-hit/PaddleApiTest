import paddle
import numpy as np

ranks = 7
dtype = "float32"
shape = [1, 192]
data_list = [np.random.random(size=shape).astype(dtype) for _ in range(ranks)]
for (i, data) in enumerate(data_list):
    np.save(f"/workspace/PaddleApiTest/test__c_concat/data/inputs_{i}.npy", data)