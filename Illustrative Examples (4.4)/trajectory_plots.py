from scipy import integrate, optimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime

T_MAX = 500
MAX_STEP = 0.01

p_l = 0.1
p_h = 0.2

M = 0.5
C = lambda p: 10 * M * (0.2 - p) ** 1 # i.e. M = 0.5, half the intervention cost
#C = lambda p: 100 * M * (0.2-p)**2
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

    return integrate.solve_ivp(grad, [0, tau], [N, y0, g1_eq, g2_eq], max_step=MAX_STEP, events=[empty_ward, empty_orbit])


def get_line(tau):
    x1, y1 = solve_trajectory_tau(35, tau).y[:2,-1]
    x2, y2 = solve_trajectory_tau(32, tau).y[:2,-1]
    
    slope = (y1-y2)/(x2-x1)
    y_int = y1 + (x1-N)*slope
    x_int = x2 + y2/slope
    
    return x_int, y_int
    

def plot_trajectory(i,j=-1):
    points = np.vstack(states_list[i:j]).T
    plt.scatter(points[0], points[1], s=0.1)
    

def trace_trajectory(y0):
    sol = solve_trajectory(y0)
    xs = sol.y[0]
    ys = sol.y[1]
    plt.scatter(xs, ys, s=0.1, label=y0)
    np.vectorize(policy)(sol.y[-1])


y0s = np.concatenate([
        np.linspace(0, (mu * N - Lambda) / nu, 1000),
        np.linspace(42, 44, 200),
        np.linspace(35.5, 36.5, 100)
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


x_dim = 4*140+1
y_dim = 4*140+1
x_grid = np.linspace(0, 140, x_dim)
y_grid = np.linspace(0, 140, y_dim)
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


def simulate_trajectory(initial_state, knn, T_MAX=200, MAX_STEP=0.005):
    def empty_state(t, vec):
        return min(vec)  # Stop the integration when either x or y reaches zero

    empty_state.terminal = True
    empty_state.direction = -1

    def grad(t, vec):
        x, y = vec
        p = knn.predict([[x, y]])[0]  # get the optimal policy

        dx = Lambda + nu * y - mu * min(x, N)
        dy = -nu * y + mu * min(x, N) * p

        return np.array([dx, dy])

    return integrate.solve_ivp(grad, [0, T_MAX], initial_state, max_step=MAX_STEP, events=empty_state)

# Trajectory plot

from matplotlib import rcParams
def plot_multiple_trajectories(initial_states, knn):
    plt.figure(figsize=(6,6))  # Adjust size as needed
    light_red = "#FFCCCC"
    rcParams['font.size'] = 14  # Adjust as needed
    # 4. Predict using both models to get the full decision boundary
    x_vals_left = np.linspace(min(x_boundary_1), max(x_boundary_1), 100)
    x_vals_right = np.linspace(min(x_boundary_2), max(x_boundary_2), 100)

    y_vals_left = poly_regressor.predict(poly_transformer.transform(x_vals_left.reshape(-1, 1)))
    y_vals_right = linear_regressor.predict(x_vals_right.reshape(-1, 1))

    for initial_state in initial_states:
        # Simulate the trajectory for the given initial condition
        sol = simulate_trajectory(initial_state, knn)
        xs = sol.y[0]
        ys = sol.y[1]

        # Plot the trajectory in the 2D space
        plt.plot(xs, ys, label=fr'$({initial_state[0]}, {initial_state[1]})$')

    plt.axvline(50, color='grey', linestyle='dotted')  # Vertical dotted line at x=50
    plt.plot([0, 50], [45, 45], color='grey', linestyle='dotted')  # Horizontal dotted line from (0,60) to (50,60)
    #plt.imshow(policy_grid.T, extent=[0, 120, 0, 120], origin='lower', cmap='viridis', alpha=0.9)
    #plt.colorbar(label='Policy Value')
    
    plt.xlim(0, 120)  
    plt.ylim(0, 120)  
    plt.xlabel('Needy State (x)')
    plt.ylabel('Content State (y)')
    #plt.title('2D Trajectories under KNN Policy')
    plt.xlim(left=0)  # Setting x-axis to start from 0
    plt.ylim(bottom=0)  # Setting y-axis to start from 0
    plt.plot(x_vals_left, y_vals_left, '--k')
    plt.plot(x_vals_right, y_vals_right, '--k')
    plt.legend()
    plt.show()

initial_states = [[20,100], [20,80], [5,65]]  # List of initial conditions
plot_multiple_trajectories(initial_states, knn)  

# Stream plot

def compute_grad(x, y, g1, g2):
    p = knn.predict([[x, y]])[0]
    dx = Lambda + nu * y - mu * min(x, N)
    dy = -nu * y + mu * min(x, N) * p
    return dx, dy, None, None  # We only care about dx and dy for the stream plot

# Define the grid of points where you want to evaluate the vector field
x_vals = np.linspace(0, 120, 120)
y_vals = np.linspace(0, 120, 120)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)

# Define the vector field at each point
u_grid = np.zeros_like(x_grid)
v_grid = np.zeros_like(y_grid)
for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        x, y = x_grid[i, j], y_grid[i, j]
        g2 = g2_eq  # Assuming you want to visualize under the equilibrium value of g2
        u, v, _, _ = compute_grad(x, y, g1_eq, g2_eq)
        u_grid[i, j] = u
        v_grid[i, j] = v

# Create the stream plot
plt.figure(figsize=(6, 6))  # Set the figure size
plt.streamplot(x_vals, y_vals, u_grid, v_grid, density=1.5)
plt.axvline(50, color='grey', linestyle='dotted')  # Vertical dotted line at x=50
plt.plot([0, 50], [45, 45], color='grey', linestyle='dotted')  # Horizontal dotted line from (0,60) to (50,60)
plt.plot(x_vals_left, y_vals_left, '--k')
plt.plot(x_vals_right, y_vals_right, '--k')
plt.xlabel('Needy State (x)')
plt.ylabel('Content State (y)')
#plt.title('Optimal Policy Dynamics')
plt.show()


# Define the grid over which to evaluate the KNN policy
x_dim, y_dim = 120, 120
x_grid = np.linspace(0, 120, x_dim) # Adjust range as needed
y_grid = np.linspace(0, 120, y_dim) # Adjust range as needed
xy_grid = np.array([[(x, y) for y in y_grid] for x in x_grid])

# Get the KNN policy predictions for each point in the grid
inputs = xy_grid.reshape(-1, 2)
preds = knn.predict(inputs)
policy_grid = preds.reshape(x_dim, y_dim)

# Plot the policy as a 2D image
fig, ax = plt.subplots(figsize=(12.5, 6))
img = ax.imshow(policy_grid.T, origin='lower', extent=[0, 120, 0, 120], aspect='auto', cmap='viridis')

# Add labels, title, and colorbar
ax.set_xlabel('Needy State (x)')
ax.set_ylabel('Content State (y)')
ax.set_title('KNN Policy over Two Dominions')
fig.colorbar(img, label='Policy Value')

plt.show()




