import numpy as np

card_num = 8

def generate_np_inputs_and_dout():
    np.random.seed(0)
    logits = np.random.random(size=[1, 46256]).astype("float32")-0.5
    label = np.random.randint(46256, size=[1]).astype("int64")
    dout = np.random.random(size=[1]).astype("float32")-0.5

    np.savez("./inputs_case1.npz", logits=logits, label=label, dout=dout)

    logits = np.random.random(size=[8192, 31250 * card_num]).astype("float32")-0.5
    label = np.random.randint(31250 * card_num, size=[8192]).astype("int64")
    dout = np.random.random(size=[8192]).astype("float32")-0.5

    np.savez("./inputs_case2.npz", logits=logits, label=label, dout=dout)