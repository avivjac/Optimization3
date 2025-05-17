import numpy as np

# --------------a--------------
#Jacobi method implementation
def jacobi(A, b, x0, max_iter=100, omega=1.0):
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(D)
    x = x0.copy()
    errors = []
    conv_factors = []

    for k in range(max_iter):
        x_new = x + omega * (D_inv @ (b - A @ x))
        error = np.linalg.norm(A @ x_new - b)
        errors.append(error)
        if k > 0:
            conv_factors.append(error / errors[k - 1])
        x = x_new
    return x, errors, conv_factors

#gauss_seidel method implementation
def gauss_seidel(A, b, x0, max_iter=100):
    L = np.tril(A)
    U = A - L
    x = x0.copy()
    errors = []
    conv_factors = []

    for k in range(max_iter):
        x_new = np.linalg.solve(L, b - U @ x)
        error = np.linalg.norm(A @ x_new - b)
        errors.append(error)
        if k > 0:
            conv_factors.append(error / errors[k - 1])
        x = x_new
    return x, errors, conv_factors

#steepest_descent method implementation
def steepest_descent(A, b, x0, max_iter=100):
    x = x0.copy()
    errors = []
    conv_factors = []

    for k in range(max_iter):
        r = b - A @ x
        alpha = r @ r / (r @ A @ r)
        x = x + alpha * r
        error = np.linalg.norm(A @ x - b)
        errors.append(error)
        if k > 0:
            conv_factors.append(error / errors[k - 1])
    return x, errors, conv_factors

#conjugate_gradient method implementation
def conjugate_gradient(A, b, x0, max_iter=100):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    errors = []
    conv_factors = []

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        error = np.linalg.norm(A @ x - b)
        errors.append(error)
        if k > 0:
            conv_factors.append(error / errors[k - 1])
        if np.sqrt(rs_new) < 1e-10:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, errors, conv_factors

# --------------b--------------
import scipy.sparse as sparse
import matplotlib.pyplot as plt

# create a random sparse matrix A
n = 256
A_rand = sparse.random(n, n, density=5/n, format='csr')
v = np.random.rand(n)
V = sparse.spdiags(v, 0, n, n, format='csr')
A = A_rand.transpose() @ V @ A_rand + 0.1 * sparse.eye(n)
A = A.toarray() 

b = np.random.randn(n)
x0 = np.zeros(n)

# Run the methods
x_jacobi, err_jacobi, conv_jacobi = jacobi(A, b, x0, omega=0.1)
x_gs, err_gs, conv_gs = gauss_seidel(A, b, x0)
x_sd, err_sd, conv_sd = steepest_descent(A, b, x0)
x_cg, err_cg, conv_cg = conjugate_gradient(A, b, x0)

errors = [
    (err_jacobi, "Jacobi"),
    (err_gs, "Gauss-Seidel"),
    (err_sd, "Steepest Descent"),
    (err_cg, "Conjugate Gradient")
]

convs = [
    (conv_jacobi, "Jacobi"),
    (conv_gs, "Gauss-Seidel"),
    (conv_sd, "Steepest Descent"),
    (conv_cg, "Conjugate Gradient")
]

# plot the errors
for err, name in errors:
    plt.figure(figsize=(8, 5))
    plt.semilogy(err)
    plt.title(f"Residual Norm: ||Ax(k) - b|| for {name}")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot the convergence factors
for conv, name in convs:
    plt.figure(figsize=(8, 5))
    plt.plot(conv)
    plt.title(f"Convergence Factor for {name}")
    plt.xlabel("Iteration")
    plt.ylabel("||Ax(k) - b|| / ||Ax(k-1) - b||")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------c--------------

At = A.T
A_ls = At @ A
b_ls = At @ b
x0 = np.zeros_like(b)

# solving LS problem using the same methods
x_jacobi_ls, err_jacobi_ls, conv_jacobi_ls = jacobi(A_ls, b_ls, x0, omega=0.6)
x_gs_ls, err_gs_ls, conv_gs_ls = gauss_seidel(A_ls, b_ls, x0)
x_sd_ls, err_sd_ls, conv_sd_ls = steepest_descent(A_ls, b_ls, x0)
x_cg_ls, err_cg_ls, conv_cg_ls = conjugate_gradient(A_ls, b_ls, x0)


errors_ls = [
    (err_jacobi_ls, "Jacobi (LS)"),
    (err_gs_ls, "Gauss-Seidel (LS)"),
    (err_sd_ls, "Steepest Descent (LS)"),
    (err_cg_ls, "Conjugate Gradient (LS)")
]

convs_ls = [
    (conv_jacobi_ls, "Jacobi (LS)"),
    (conv_gs_ls, "Gauss-Seidel (LS)"),
    (conv_sd_ls, "Steepest Descent (LS)"),
    (conv_cg_ls, "Conjugate Gradient (LS)")
]

# plotting the errors for LS problem
for err, name in errors_ls:
    plt.figure(figsize=(8, 5))
    plt.semilogy(err)
    plt.title(f"Residual Norm: ||Ax(k) - b|| for {name}")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plotting the convergence factors for LS problem
for conv, name in convs_ls:
    plt.figure(figsize=(8, 5))
    plt.plot(conv)
    plt.title(f"Convergence Factor for {name}")
    plt.xlabel("Iteration")
    plt.ylabel("||Ax(k) - b|| / ||Ax(k-1) - b||")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

np.allclose(x_cg, x_cg_ls, atol=1e-4)