import numpy as np

if __name__ == "__main__":
    # TODO: dummy input data. The output data corresponds to the result from the input 
    # data generated with a given random seed (the one used for the experimentation in 
    # the paper
    in_data = np.array([[0.0] * 1 * 3 * 4 * 4], dtype=np.float32)
    out_data = np.array([[-0.05242537, -0.28872108]], dtype=np.float32)

    with open("input.npy", "wb") as f_in:
        np.save(f_in, in_data)
        
    with open("expected_output.npy", "wb") as f_out:
        np.save(f_out, out_data)
