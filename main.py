import numpy as np
import matplotlib.pyplot as plt

iterations_limit = 10000

def calculate_objective(x):
    x1 = x[0]
    x2 = x[1]

    f_x = 100 * (x2 - (x1 ** 2)) ** 2 + (1 - x1) ** 2
    return f_x

def compute_gradient(x):
    x1 = x[0]
    x2 = x[1]

    grad_x1 = 400 * np.power(x1, 3) - 400 * x1 * x2 + 2 * x1 - 2
    grad_x2 = -200 * np.power(x1, 2) + 200 * x2

    return np.array([grad_x1, grad_x2])

def compute_hessian(x):
    x1 = x[0]
    x2 = x[1]

    a11 = 1200 * (x1 ** 2) - 400 * x2 + 2
    a12 = -400 * x1
    a21 = a12
    a22 = 200

    return np.array([[a11, a12], [a21, a22]])

def hessian_bfgs(H_k, delta_grad, delta_x):

    a = 1 + (np.dot(((delta_grad.T) @ H_k), delta_grad) / np.dot((delta_grad.T), delta_x))
    b = (np.outer(delta_x, (delta_x.T))) / (np.dot(delta_x.T, delta_grad))
    c = (np.outer((H_k @ delta_grad), delta_x.T) + (np.outer((H_k @ delta_grad), delta_x.T).T)) / (np.dot((delta_grad.T), delta_x))

    H_next = H_k + a*b - c
    return H_next

def newton(max_iterations, x0):
    error = 1e-6

    func_arr = []
    d_arr = []
    hessian_arr = []
    inverse_hessian_arr = []
    error_arr = []

    x_k = x0

    for i in range(max_iterations):
        func_arr.append(calculate_objective(x_k).copy())

        gradient = compute_gradient(x_k) * -1

        hessian = compute_hessian(x_k)
        hessian_arr.append(hessian.copy())
        inverse_hessian_arr.append(np.linalg.inv(hessian).copy())

        d_x = np.linalg.solve(hessian, gradient)
        d_arr.append(d_x.copy())

        error_x = x_k
        x_k = x_k + d_x

        error_x = np.linalg.norm((x_k - error_x), 2)
        error_arr.append(error_x)

        if (error_x < error):
            print(f"Converged at step {i}")
            print(f"x: {x_k}, {func_arr[i]}")
            return np.array(func_arr), np.array(d_arr), np.array(hessian_arr), np.array(inverse_hessian_arr), np.array(error_arr)

    print(f"Stopped after {max_iterations} iterations")
    print(f"x: {x_k}, {func_arr[-1]}")
    return np.array(func_arr), np.array(d_arr), np.array(hessian_arr), np.array(inverse_hessian_arr), np.array(error_arr)


def quasi_newton(max_iterations, x0):
    error = 1e-6

    func_arr = []
    d_arr = []
    H_arr = []
    inverse_hessian_arr = []
    error_arr = []

    x_k = x0
    x_prev = x0
    H_k = np.eye(2)
    H_arr.append(H_k.copy())
    for i in range(max_iterations):
        func_arr.append(calculate_objective(x_k).copy())

        gradient = compute_gradient(x_k) * -1
        inverse_hessian_arr.append(np.linalg.inv(compute_hessian(x_k)).copy())

        d_x = np.dot(H_k, gradient)
        d_arr.append(d_x.copy())

        x_prev = x_k
        x_k = x_k + d_x

        error_x = np.linalg.norm((x_k - x_prev), 2)
        error_arr.append(error_x)

        if (np.abs(calculate_objective(x_k) - func_arr[i]) < error):
            print(f"Converged at step {i}")
            print(f"x: {x_k}, {func_arr[i]}")
            return np.array(func_arr), np.array(d_arr), np.array(H_arr), np.array(inverse_hessian_arr), np.array(error_arr)

        if (i != max_iterations - 1):
            H_k = hessian_bfgs(H_k, (compute_gradient(x_k) + gradient), (x_k - x_prev))
            H_arr.append(H_k.copy())

    print(f"Stopped after {max_iterations} iterations")
    print(f"x: {x_k}, {func_arr[-1]}")
    return np.array(func_arr), np.array(d_arr), np.array(H_arr), np.array(inverse_hessian_arr), np.array(error_arr)

def main():
    # Task 1
    x0_1 = np.array([2, 4])
    x0_2 = np.array([-2, 10])

    fig, ax = plt.subplots()
    fig_error, ax_error = plt.subplots()

    func_arr, d_arr, hessian_arr, inverse_hessian_arr, error_arr = newton(iterations_limit, x0_1)
    ax.plot([i for i in range(0, len(func_arr))], func_arr, label=x0_1)
    ax_error.plot([i for i in range(1, len(error_arr)+1)], error_arr, label=x0_1)

    func_arr, d_arr, hessian_arr, inverse_hessian_arr, error_arr = newton(iterations_limit, x0_2)
    ax.plot([i for i in range(0, len(func_arr))], func_arr, label=x0_2)
    ax_error.plot([i for i in range(1, len(error_arr)+1)], error_arr, label=x0_2)

    ax_error.set_yscale('log')
    ax.legend()
    ax_error.legend()
    plt.show()

    # Task 2
    fig, ax = plt.subplots()
    fig_error, ax_error = plt.subplots()
    fig_apr, ax_apr = plt.subplots()

    func_arr, d_arr, H_arr, inverse_hessian_arr, error_arr = quasi_newton(iterations_limit, x0_1)
    ax.plot([i for i in range(0, len(func_arr))], func_arr, label=x0_1)
    ax_error.plot([i for i in range(1, len(error_arr) + 1)], error_arr, label=x0_1)
    ax_apr.plot([i for i in range(1, len(H_arr) + 1)], [np.linalg.norm((inverse_hessian_arr[i] - H_arr[i]),2) for i in range(0, len(H_arr))], label=x0_1)

    func_arr, d_arr, H_arr, inverse_hessian_arr, error_arr = quasi_newton(iterations_limit, x0_2)
    ax.plot([i for i in range(0, len(func_arr))], func_arr, label=x0_2)
    ax_error.plot([i for i in range(1, len(error_arr) + 1)], error_arr, label=x0_2)
    ax_apr.plot([i for i in range(1, len(H_arr) + 1)], [np.linalg.norm((inverse_hessian_arr[i] - H_arr[i]),2) for i in range(0, len(H_arr))], label=x0_2)

    ax_error.set_yscale('log')
    ax_apr.set_yscale('log')
    ax.legend()
    ax_error.legend()
    ax_apr.legend()
    plt.show()

    return 0

main()