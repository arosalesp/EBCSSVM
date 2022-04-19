cimport numpy as np
cimport cython
from numpy cimport ndarray as ar
from numpy import empty, zeros
from libc.stdlib cimport free

cdef extern from "math.h":
    double exp(double t)

@cython.boundscheck(False)
def rbf_kernel(ar[double, ndim=2] X, ar[double, ndim=2] Y, double gamma):
    cdef Py_ssize_t i, j, k
    cdef int n_x = X.shape[0]
    cdef int n_y = Y.shape[0]
    cdef Py_ssize_t d_x = X.shape[1]
    cdef Py_ssize_t d_y = Y.shape[1]
    cdef ar[double, ndim=2] m = empty((n_x, n_y))
    cdef double tmp_1, tmp_2
    
    assert d_x == d_y
    
    for i in range(n_x):
        for j in range(n_y):
            tmp_2 = 0
            for k in range(d_x):
                tmp_1 = X[i, k] - Y[j, k]
                tmp_2 += (tmp_1 * tmp_1)
            m[i, j] = exp(-gamma * tmp_2)
        
    return m

def rbf_kernel_2(ar[double, ndim=2] X, double gamma):
    cdef Py_ssize_t i, j, k
    cdef int n_x = X.shape[0]
    cdef Py_ssize_t d_x = X.shape[1]
    cdef ar[double, ndim=2] m = empty((n_x, n_x))
    cdef double tmp_1, tmp_2
    
    assert n_x == d_x
    
    for i in range(n_x):
        for j in range(n_x):
            m[i, j] = exp(-gamma * X[i, j])
        
    return m

def squared_Euclidean(ar[double, ndim=2] X, ar[double, ndim=2] Y):
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t n_x = X.shape[0]
    cdef Py_ssize_t n_y = Y.shape[0]
    cdef Py_ssize_t d_x = X.shape[1]
    cdef Py_ssize_t d_y = Y.shape[1]
    cdef ar[double, ndim=2] m = empty((n_x, n_y))
    cdef double tmp_1, tmp_2
    
    assert d_x == d_y
    
    for i in range(n_x):
        for j in range(n_y):
            tmp_2 = 0
            for k in range(d_x):
                tmp_1 = X[i, k] - Y[j, k]
                tmp_2 += (tmp_1 * tmp_1)
            m[i, j] = tmp_2
        
    return m
            
cpdef smo(ar[double, ndim=2] Q, ar[long, ndim=1] y, ar[double, ndim=1] alpha,
          ar[double, ndim=1] p, ar[double, ndim=1] bias,
          ar[double, ndim=1] box, int max_iter, ar[double, ndim=1] cost_val):
    cdef int i, j, iter_num, k
    cdef int n_x = Q.shape[0]
    cdef int n_y = y.shape[0]
    cdef Py_ssize_t d_x = Q.shape[1]
    cdef ar[double, ndim=1] grad = zeros((n_y))
    cdef double m_alpha, M_alpha, sum_alpha, Gmax, tmp_1, aij, bij, obj, diff, suma
    cdef double delta_alpha_i, delta_alpha_j, old_alpha_i, old_alpha_j, eta, delta
    cdef double pp, pm, factor
    
    assert n_x == n_y
    
    m_alpha = -1e32
    M_alpha = -1e32
    Gmax = -1e32
    
    cost_val[0] = 0
    sum_alpha = 0

    for i in range(n_x):
        for j in range(i, n_x):
            grad[i] += alpha[i] * Q[i, j]
            cost_val[0] += (alpha[i] * alpha[j] * Q[i, j])
        grad[i] -= p[i]
        sum_alpha += alpha[i]
        if (y[i] == 1) and (alpha[i] < box[i]):
            if -grad[i] >= m_alpha:
                m_alpha = -grad[i]
        elif (y[i] == -1) and (alpha[i] > 0):
            if grad[i] >= m_alpha:
                m_alpha = grad[i]
        if (y[i] == 1) and (alpha[i] > 0):
            if grad[i] >= M_alpha:
                M_alpha = grad[i]
        elif (y[i] == -1) and (alpha[i] < box[i]):
            if -grad[i] >= M_alpha:
                M_alpha = -grad[i]
        
    cost_val[0] *= 0.5
    cost_val[0] -= sum_alpha
    iter_num = 0
    while ((m_alpha + M_alpha) > 1e-6) and (iter_num < max_iter):
        iter_num = iter_num + 1
        m_alpha = -1e32
        M_alpha = -1e32
        Gmax = 1e32
        i = -1
        j = -1
        # Chossing i and j
        for k in range(n_x):
            if (y[k] == 1) and (alpha[k] < box[k]):
                if -grad[k] >= m_alpha:
                    m_alpha = -grad[k]
                    i = k
            elif (y[k] == -1) and (alpha[k] > 0):
                if grad[k] >= m_alpha:
                    m_alpha = grad[k]
                    i = k
        if i == -1:
            break
        for k in range(n_x):
            if (y[k] == 1) and (alpha[k] > 0):
                bij = m_alpha + grad[k]
                if grad[k] >= M_alpha:
                    M_alpha = grad[k]
                if bij > 0:
                    aij = Q[i, i] + Q[k, k] - 2 * y[i] * Q[i, j]
                    if aij <= 0:
                        aij = 1e-12
                    obj = -(bij * bij) / aij
                    if obj <= Gmax:
                        Gmax = obj
                        j = k
            elif (y[k] == -1) and (alpha[k] < box[k]):
                bij = m_alpha - grad[k]
                if -grad[k] >= M_alpha:
                    M_alpha = -grad[k]
                if bij > 0:
                    aij = Q[i, i] + Q[k, k] + 2 * y[i] * Q[i, j]
                    if aij <= 0:
                        aij = 1e-12
                    obj = -(bij * bij) / aij
                    if obj <= Gmax:
                        Gmax = obj
                        j = k
        if (j == -1):
            break
        
        old_alpha_i = alpha[i]
        old_alpha_j = alpha[j]
        if y[i] != y[j]:
            eta = Q[i, i] + Q[k, k] + 2 * Q[i, j]
            # eta *= factor
            if eta <= 0:
                eta = 1e-12
            delta = (-grad[i] - grad[j]) / eta
            diff = alpha[i] - alpha[j]
            alpha[i] += delta
            alpha[j] += delta
            if diff > 0:
                if alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = diff
            else:
                if alpha[i] < 0:
                    alpha[i] = 0
                    alpha[j] = -diff
            if diff > (box[i]- box[j]):
                if alpha[i] > box[i]:
                    alpha[i] = box[i]
                    alpha[j] = box[i] - diff
            else:
                if alpha[j] > box[j]:
                    alpha[j] = box[j]
                    alpha[i] = box[j] + diff
        else:
            eta = Q[i, i] + Q[k, k] - 2 * Q[i, j]
            # eta *= factor
            if eta <= 0:
                eta = 1e-12
            delta = (grad[i] - grad[j]) / eta
            suma = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta
            if suma > box[i]:
                if alpha[i] > box[i]:
                    alpha[i] = box[i]
                    alpha[j] = suma - box[i]
            else:
                if alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = suma
            if suma >  box[j]:
                if alpha[j] > box[j]:
                    alpha[j] = box[j]
                    alpha[i] = suma - box[j]
            else:
                if alpha[i] < 0:
                    alpha[i] = 0
                    alpha[j] = suma
        
        delta_alpha_i = alpha[i] - old_alpha_i
        delta_alpha_j = alpha[j] - old_alpha_j
        
        for k in range(n_x):
            grad[k] += (delta_alpha_i * Q[i, k] + delta_alpha_j * Q[j, k])
        
            cost_val[0] += (delta_alpha_i * Q[i, k] + delta_alpha_j * Q[j, k])
        cost_val[0] += (delta_alpha_i * delta_alpha_j * Q[i, j] + 
                        0.5 * delta_alpha_i * delta_alpha_i * Q[i, i] +
                        0.5 * delta_alpha_j * delta_alpha_j * Q[j, j] - 
                        delta_alpha_i - delta_alpha_j)
        sum_alpha += (delta_alpha_i + delta_alpha_j)

            
    cdef double ub = 1e32
    cdef double lb = -1e32
    cdef double sum_free = 0
    cdef double y_grad
    cdef int num_free = 0
    
    for k in range(n_x):
        y_grad = y[i] * grad[i]
        
        if alpha[i] == box[i]:
            if y[i] == -1:
                ub = min(ub, y_grad)
            else:
                lb = max(lb, y_grad)
        elif alpha[i] == 0:
            if y[i] == 1:
                ub = min(ub, y_grad)
            else:
                lb = max(lb, y_grad)
        else:
            num_free += 1
            sum_free -= y_grad
    
    if num_free > 0:
        bias[0] = sum_free / num_free
    else:
        bias[0] = (-ub - lb) / 2












