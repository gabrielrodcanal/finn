import numpy as np

if __name__ == "__main__":
    # TODO: dummy input data. The output data corresponds to the result from the input 
    # data generated with a given random seed (the one used for the experimentation in 
    # the paper
    in_data = np.array([[0.0] * 1 * 3 * 10 * 10], dtype=np.float32)
    out_data = np.array([[ 0.0063, -0.1619,  0.0774, -0.1374,  0.0056, -0.1590, -0.0842, -0.0527,
                 -0.0687, -0.1741]], dtype=np.float32)
    with open("input.npy", "wb") as f_in:
        np.save(f_in, in_data)
        
    with open("expected_output.npy", "wb") as f_out:
        np.save(f_out, out_data)
