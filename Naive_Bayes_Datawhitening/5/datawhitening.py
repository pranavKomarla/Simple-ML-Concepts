import numpy as np



if __name__ == "__main__":

    # 5.1

    X=np.load("x.npz")["x"]
  

    # First we want to compute the mean of X which would equal E[X]

    mu = np.mean(X, axis=1)
    
    
    N = X.shape[1] # getting the number of samples

    # We want to compute the covariance matrix of X which is COV[X, X]

    
    X_centered = X - mu[:, np.newaxis] # We want to cetner the data by subtracting the mean from each individual sample

    COV_X = np.matmul(X_centered, X_centered.T) / (N - 1)

    print("\nMean (E[X]):\n", mu)
    print("\nCovariance Matrix COV[X, X]: \n", COV_X)

    # 5.2 Let W be a random vector defined by W = AX + b. Express E[W] and COV [W, W] in terms of E[X] = μ and COV [X, X] = Σ

    # We will be performing SVD on the covariance matrix, in order to find the whitening matrix of A -> A = S^(-1/2)*U.T
    
    U, S, Vt = np.linalg.svd(COV_X)
    print("\nThe Eigenvalues are: \n", S)
    print("\nThe EigenVectors are: \n", U)

    # With the eigenvalues and eigenvectors we can compute the resulting whitening matrix A
    A = np.matmul(np.diag(1.0/np.sqrt(S)), U.T)
    print("\nThe Whitening Matrix (A) is: ", A)

    # 5.3 Now we can solve for E[W] and COV[W, W]

    """
        Now that we have the whitening matrix, we can solve for b, which then allows us to solve for W
        W = AX + b

        b = -Au
    """

    b = -np.matmul(A, mu)
    print("\nThe bias Vector (B) is: ", b)

    """
        Now we can solve for E[W] and COV[W, W]

        E[W] = E[AX + b] = AE[X] + b = Au + b 

        COV[W, W] = COV[AX+b, AX+b]

        covariance is not affect by constant shifts, therefore b will disappear
        COV[W,W] = ACOV[X,X]A^T = A * Σ * A^T

        E[W] = 0, COV[W, W] = I
    """

    # Compute E[W] 
    E_W = np.matmul(A, mu) + b
    print("\nE[W]: This should just be zeroes\n", E_W)

    # Compute the covariance of the this new transformed data
    COV_WW = np.matmul(np.matmul(A, COV_X), A.T)
    print("\nCovariance matrix: This should be an Identity Matrix:\n", COV_WW)




