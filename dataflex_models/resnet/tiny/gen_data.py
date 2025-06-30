import numpy as np

if __name__ == "__main__":
    # TODO: dummy input data. The output data corresponds to the result from the input 
    # data generated with a given random seed (the one used for the experimentation in 
    # the paper
    in_data = np.array([[0.0] * 1 * 3 * 3 * 3], dtype=np.float32)
    out_data = np.array([[ 0.21210718, -0.4147637,  0.28054863,  0.3298071,  -0.3468396,  -0.34683585,
          -0.0963487,  -0.06894457,  0.3090791,  -0.21088159]], dtype=np.float32)

    with open("input.npy", "wb") as f_in:
        np.save(f_in, in_data)
        
    with open("expected_output.npy", "wb") as f_out:
        np.save(f_out, out_data)
