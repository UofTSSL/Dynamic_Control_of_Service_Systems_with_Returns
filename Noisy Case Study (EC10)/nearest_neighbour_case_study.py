# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:52:47 2021

@author: Simon
"""

from scipy import integrate, optimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

T_MAX = 300
MAX_STEP = 0.02


hr_full = 0.62
hr_basic = 0.89

# p_h = 0.104
# p_mid = hr_basic * p_h
# p_l = hr_full * p_h

p_h = 0.141
p_l = 0.081

# hourly_rate = 150
# M_full = hourly_rate * 114/60
# M_basic = hourly_rate * 26/60

M = 1110

def C(p):
    return M * (p_h - p) / (p_h - p_l)
    
    # if p < p_mid:
    #     a = (p_mid - p) / (p_mid - p_l)
    #     return a * (M_full - M_basic) + M_basic
    #     
    # a = (p_h - p) / (p_h - p_mid)
    # return a * M_basic

r = 5000.0
h = 500.0

mu = 1 / 5.6
nu = 1 / 12.07
Lambda = 206.3/30
N = 45


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
    empty_orbit.terminal = -1.

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

y0s = np.concatenate([
        np.linspace(0, (mu * N - Lambda) / nu, 200)
        #np.linspace(0, 32, 150),
        #np.linspace(32, 33, 100),
        #np.linspace(33, 38, 100),
        #np.linspace(38, 52.5, 100),
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

knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(all_states, all_ps)

x_dim = 161
y_dim = 161
x_grid = np.linspace(0, 80, x_dim)
y_grid = np.linspace(0, 80, y_dim)
xy_grid = np.array([[(x, y) for y in y_grid] for x in x_grid])


inputs = xy_grid.reshape(-1, 2)
preds = knn.predict(inputs)
outputs = preds.reshape(x_dim, y_dim)


fig, ax = plt.subplots(1, 1)
img = ax.pcolor(x_grid, y_grid, outputs.T)
# img = ax.imshow(outputs.T, origin='lower', interpolation='nearest', aspect='auto')
ax.set_xlabel('Number of Patients in Ward (x)')
ax.set_ylabel('Number of Patients in Orbit (y)')
ax.set_title('Optimal Policy (p)')
ax.axvline(N, linewidth=1, linestyle='dotted')
fig.colorbar(img)
plt.savefig('paper_lin_policy.png', bbox_inches='tight')

def plot_trajectory(i,j=-1):
    points = np.vstack(states_list[i:j]).T
    plt.scatter(points[0], points[1], s=0.1)
    
# states = np.array(states_list)
# np.save("states_linear", states)
# ps = np.array(ps_list)
# np.save("ps_linear", ps)
