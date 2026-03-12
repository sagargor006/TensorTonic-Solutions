import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    # print(f"input: {A}")
    A = np.array(A)
    # print(f"input: {A}")
    rows = A.shape[1]
    columns = A.shape[0]

    # print(f"shape: {A.shape} rows: {rows}, columns: {columns}")
    A_T = np.zeros((rows,columns))
    for i in range(rows):
        for j in range(columns):
            A_T[i,j] = A[j,i]
            # print(f"A_T[{i},{j}] = A[{j},{i}]")
    return A_T
        
