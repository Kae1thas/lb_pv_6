from mpi4py import MPI
from numpy import empty, array, zeros, int32, float64, arange, dot, sqrt
from matplotlib.pyplot import style, figure, axes, show
import time

def conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, N, comm_cart, num_row, num_col):
    neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)
    neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)

    r_part = empty(N_part, dtype=float64)
    p_part = empty(N_part, dtype=float64)
    q_part = empty(N_part, dtype=float64)
    ScalP_temp = empty(1, dtype=float64)
    rsold = array(0.0, dtype=float64)

    p_part = zeros(N_part, dtype=float64)
    max_iter = 50  # Для теста
    eps = 1e-6

    # r = b - A x
    Ax_part_temp = dot(A_part, x_part)
    Ax_part = Ax_part_temp.copy()
    for n in range(num_col - 1):
        comm_cart.Sendrecv_replace([Ax_part_temp, M_part, MPI.DOUBLE],
                                   dest=neighbour_right, sendtag=0,
                                   source=neighbour_left, recvtag=MPI.ANY_TAG)
        Ax_part += Ax_part_temp
    residual_part = b_part - Ax_part

    r_part_temp = dot(A_part.T, residual_part)
    r_part = r_part_temp.copy()
    for m in range(num_row - 1):
        comm_cart.Sendrecv_replace([r_part_temp, N_part, MPI.DOUBLE],
                                   dest=neighbour_down, sendtag=0,
                                   source=neighbour_up, recvtag=MPI.ANY_TAG)
        r_part += r_part_temp

    # rsold = (r, r)
    ScalP_temp[0] = dot(r_part, r_part)
    rsold = ScalP_temp.copy()
    for n in range(num_col - 1):
        comm_cart.Sendrecv_replace([ScalP_temp, 1, MPI.DOUBLE],
                                   dest=neighbour_right, sendtag=0,
                                   source=neighbour_left, recvtag=MPI.ANY_TAG)
        rsold += ScalP_temp
    if sqrt(rsold[0]) < eps:
        return x_part

    p_part = r_part.copy()

    for s in range(1, max_iter + 1):
        # q = A p
        Ap_part_temp = dot(A_part, p_part)
        Ap_part = Ap_part_temp.copy()
        for n in range(num_col - 1):
            comm_cart.Sendrecv_replace([Ap_part_temp, M_part, MPI.DOUBLE],
                                       dest=neighbour_right, sendtag=0,
                                       source=neighbour_left, recvtag=MPI.ANY_TAG)
            Ap_part += Ap_part_temp
        q_part_temp = dot(A_part.T, Ap_part)
        q_part = q_part_temp.copy()
        for m in range(num_row - 1):
            comm_cart.Sendrecv_replace([q_part_temp, N_part, MPI.DOUBLE],
                                       dest=neighbour_down, sendtag=0,
                                       source=neighbour_up, recvtag=MPI.ANY_TAG)
            q_part += q_part_temp

        # alpha = (r,r) / (p, q)
        ScalP_temp[0] = dot(p_part, q_part)
        alpha_denom = ScalP_temp.copy()
        for n in range(num_col - 1):
            comm_cart.Sendrecv_replace([ScalP_temp, 1, MPI.DOUBLE],
                                       dest=neighbour_right, sendtag=0,
                                       source=neighbour_left, recvtag=MPI.ANY_TAG)
            alpha_denom += ScalP_temp
        alpha = rsold[0] / alpha_denom[0]

        x_part += alpha * p_part
        r_part -= alpha * q_part

        # rsnew = (r, r)
        ScalP_temp[0] = dot(r_part, r_part)
        rsnew = ScalP_temp.copy()
        for n in range(num_col - 1):
            comm_cart.Sendrecv_replace([ScalP_temp, 1, MPI.DOUBLE],
                                       dest=neighbour_right, sendtag=0,
                                       source=neighbour_left, recvtag=MPI.ANY_TAG)
            rsnew += ScalP_temp
        if sqrt(rsnew[0]) < eps:
            break

        beta = rsnew[0] / rsold[0]
        p_part = r_part + beta * p_part
        rsold = rsnew

    return x_part

# Основная программа
comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if not sqrt(numprocs).is_integer():
    if rank == 0:
        print(f"Ошибка: numprocs={numprocs} не квадрат! Используйте 4, 9, 16...")
    comm.Abort()
    exit()

num_row = num_col = int(sqrt(numprocs))
comm_cart = comm.Create_cart(dims=(num_row, num_col), periods=(True, True), reorder=True)
rank_cart = comm_cart.Get_rank()

def auxiliary_arrays_determination(size, num_div):
    rcounts = array([size // num_div] * num_div, dtype=int32)
    if size % num_div != 0:
        rcounts[:size % num_div] += 1
    displs = array([sum(rcounts[:i]) for i in range(num_div)], dtype=int32)
    return rcounts, displs

if rank_cart == 0:
    with open('in.dat', 'r') as f1:
        N = array(int32(f1.readline().strip()))
        M = array(int32(f1.readline().strip()))
    print(f"Загружены данные: N={N}, M={M}")
else:
    N = array(0, dtype=int32)
    M = array(0, dtype=int32)

comm_cart.Bcast([N, 1, MPI.INT], root=0)
comm_cart.Bcast([M, 1, MPI.INT], root=0)

rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)

M_part = array(0, dtype=int32)
N_part = array(0, dtype=int32)

m_col = comm_cart.Split(rank_cart % num_col, rank_cart)
m_row = comm_cart.Split(rank_cart // num_col, rank_cart)

# Распределение размеров (if из лекции)
if rank_cart in range(num_col):  # first row for N_part
    m_row.Scatter([rcounts_N, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)
m_col.Bcast([N_part, 1, MPI.INT], root=0)

if rank_cart in range(0, numprocs, num_col):  # first column for M_part
    m_col.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)
m_row.Bcast([M_part, 1, MPI.INT], root=0)

# A_part: тестовая единичная матрица
A_part = zeros((M_part, N_part), dtype=float64)
min_dim = min(M_part, N_part)
for i in range(min_dim):
    A_part[i, i] = 1.0

# b
if rank_cart == 0:
    with open('bData.dat', 'r') as f3:
        b = empty(M, dtype=float64)
        for j in range(M):
            b[j] = float64(f3.readline().strip())
else:
    b = None

b_part = empty(M_part, dtype=float64)  # На всех

# Bcast full b в first column
if rank_cart in range(0, numprocs, num_col):
    if rank_cart == 0:
        full_b = b
    else:
        full_b = empty(M, dtype=float64)
    m_col.Bcast([full_b, M, MPI.DOUBLE], root=0)
    b = full_b

# Scatterv b только в first column
if rank_cart in range(0, numprocs, num_col):
    m_col.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], [b_part, M_part, MPI.DOUBLE], root=0)

# Bcast b_part по строкам (на всех)
m_row.Bcast([b_part, M_part, MPI.DOUBLE], root=0)

# x = 0
if rank_cart == 0:
    x = zeros(N, dtype=float64)
else:
    x = None

x_part = zeros(N_part, dtype=float64)  # На всех

# Bcast full x в first row
if rank_cart in range(num_col):
    if rank_cart == 0:
        full_x = x
    else:
        full_x = zeros(N, dtype=float64)
    m_row.Bcast([full_x, N, MPI.DOUBLE], root=0)
    x = full_x

# Scatterv x только в first row
if rank_cart in range(num_col):
    m_row.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], [x_part, N_part, MPI.DOUBLE], root=0)

# Bcast x_part по столбцам (на всех)
m_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)

start_time = time.time()

x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, N, comm_cart, num_row, num_col)

end_time = time.time()
if rank_cart == 0:
    print(f"Время выполнения CG: {end_time - start_time:.2f} сек")

# Gatherv x только в first row
if rank_cart == 0:
    full_x = zeros(N, dtype=float64)
else:
    full_x = None
if rank_cart in range(num_col):
    m_row.Gatherv([x_part, N_part, MPI.DOUBLE], [full_x, rcounts_N, displs_N, MPI.DOUBLE], root=0)
    if rank_cart == 0:
        x = full_x

if rank_cart == 0:
    style.use('dark_background')
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(arange(N), x, '-y', lw=3)
    show()

comm_cart.Barrier()