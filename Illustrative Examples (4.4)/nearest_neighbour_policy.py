from scipy import integrate, optimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime

T_MAX = 200
MAX_STEP = 0.005

p_l = 0.1
p_h = 0.2

# Define the cost function here
M = 0.5
# C = lambda p: 10 * M * (0.2 - p) ** 1 # i.e. M = 0.5, half the intervention cost
C = lambda p: 100 * M * (0.2-p)**2
# C = lambda p: 2*(0.2-p) if p > 0.15 else 0.1+ 8*(0.15-p) # i.e. M = 0.5, half the intervention cost

r = 1.0
h = 0.25

mu = 1 / 4
nu = 1 / 15
Lambda = 9.5
N = 50

p_eq = optimize.minimize_scalar(
    lambda q: (r * q + C(q)) / (1 - q),
    bounds=(p_l, p_h), method="bounded"
).x
g1_eq = (r * p_eq + C(p_eq)) / (1 - p_eq)
g2_eq = (r + C(p_eq)) / (1 - p_eq)


def policy(g2):
    return optimize.minimize_scalar(
        lambda q: C(q) + g2 * q,
        bounds=(p_l, p_h), method="bounded"
    ).x


def solve_trajectory(y0):
    def empty_ward(t, vec):
        return vec[0]

    empty_ward.terminal = True
    empty_ward.direction = -1.

    def empty_orbit(t, vec):
        return vec[1]

    empty_orbit.terminal = True
    empty_orbit.direction = -1.

    def grad(t, vec):
        x, y, g1, g2 = vec
        p = policy(g2)
        dx = Lambda + nu * y - mu * min(x, N)
        dy = -nu * y + mu * min(x, N) * p
        dg1 = mu * (g1 - C(p) - g2 * p) if x < N else -h
        dg2 = nu * (-r - g1 + g2)

        return -np.array([dx, dy, dg1, dg2])

    print(y0)

    return integrate.solve_ivp(grad, [0, T_MAX], [N, y0, g1_eq, g2_eq], max_step=MAX_STEP, events=[empty_ward, empty_orbit])


def solve_trajectory_tau(y0, tau):
    def empty_ward(t, vec):
        return vec[0]

    empty_ward.terminal = True
    empty_ward.direction = -1.

    def empty_orbit(t, vec):
        return vec[1]

    empty_orbit.terminal = True
    empty_orbit.direction = -1.

    def grad(t, vec):
        x, y, g1, g2 = vec
        p = policy(g2)
        dx = Lambda + nu * y - mu * min(x, N)
        dy = -nu * y + mu * min(x, N) * p
        dg1 = mu * (g1 - C(p) - g2 * p) if x < N else -h
        dg2 = nu * (-r - g1 + g2)

        return -np.array([dx, dy, dg1, dg2])

    print(y0)

    return integrate.solve_ivp(grad, [0, tau], [N, y0, g1_eq, g2_eq], max_step=MAX_STEP, events=[empty_ward, empty_orbit])


def get_line(tau):
    x1, y1 = solve_trajectory_tau(35, tau).y[:2,-1]
    x2, y2 = solve_trajectory_tau(32, tau).y[:2,-1]
    
    slope = (y1-y2)/(x2-x1)
    y_int = y1 + (x1-N)*slope
    x_int = x2 + y2/slope
    
    return x_int, y_int
    
    
for tau in [15, 35]:
    x_int, y_int = get_line(tau)
    plt.plot(np.linspace(N, x_int), np.linspace(y_int, 0), color='red', linestyle='--', linewidth=1)
    




#y0s = np.concatenate([
#        np.linspace(0, (mu * N - Lambda) / nu, 1000),
#        np.linspace(42, 44, 200),
#        np.linspace(35.5, 36.5, 100)
#])
y0s = np.concatenate([
        np.linspace(0, (mu * N - Lambda) / nu, 100),
        np.linspace(42, 44, 20),
        np.linspace(35.5, 36.5, 10)
])
sols = np.vectorize(solve_trajectory)(y0s)

states_list = []
ps_list = []

for sol in sols:
    xs = sol.y[0]
    ys = sol.y[1]
    cutoff = len(xs)
    if min(xs) < 0:
        cutoff = min((xs < 0).argmax(), cutoff)
    if min(ys) < 0:
        cutoff = min((ys < 0).argmax(), cutoff)
    ps = np.vectorize(policy)(sol.y[-1][:cutoff])

    states_list.append(np.column_stack((xs[:cutoff], ys[:cutoff])))
    ps_list.append(ps)

all_states = np.concatenate(states_list)
all_ps = np.concatenate(ps_list)


import pickle
with open('states_linear_{}.pkl'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), 'wb') as f:
    pickle.dump(states_list, f)
with open('ps_linear_{}.pkl'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), 'wb') as f:
    pickle.dump(ps_list, f)



knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(all_states, all_ps)

with open('knn_linear_{}.pkl'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), 'wb') as f:
    pickle.dump(knn, f)


x_dim = 4*4+1
y_dim = 4*4+1
x_grid = np.linspace(0, 4, x_dim)
y_grid = np.linspace(0, 4, y_dim)
xy_grid = np.array([[(x, y) for y in y_grid] for x in x_grid])

inputs = xy_grid.reshape(-1, 2)
preds = knn.predict(inputs)
outputs = preds.reshape(x_dim, y_dim)

fig, ax = plt.subplots(1, 1)
img = ax.pcolor(x_grid, y_grid, outputs.T)
# img = ax.imshow(outputs.T, origin='lower', interpolation='nearest', aspect='auto')
ax.set_xlabel(r'Needy State ($x$)')
ax.set_ylabel(r'Content State ($y$)')
ax.set_title('Optimal Policy ($p^*$)')
ax.axvline(50, linewidth=1, linestyle='dotted')
fig.colorbar(img)
plt.savefig('policy_{}.pdf'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), bbox_inches='tight')

def plot_trajectory(i,j=-1):
    points = np.vstack(states_list[i:j]).T
    plt.scatter(points[0], points[1], s=0.1)
    

def trace_trajectory(y0):
    sol = solve_trajectory(y0)
    xs = sol.y[0]
    ys = sol.y[1]
    plt.scatter(xs, ys, s=0.1, label=y0)
    np.vectorize(policy)(sol.y[-1])
        
