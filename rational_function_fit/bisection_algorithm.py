import cvxpy as cp
import numpy as np

# --- Params --- #
EPS = 1e-3	# threshold for convergence

def bisection_algorithm(t, y, alpha_l, alpha_u, max_iter):
	"""
	Implementation of bisection algorithm to solve the 
	following minimax rational fit problem:

	minimize  max |P(t_i)/Q(t_i) - y(t_i)| for all i
	    s.t.  Q(t_i) > 0                   for all i

	    where P(t) = a_0 + a_1*t + a_2*t^2
	          Q(t) =   1 + b_1*t + b_2*t^2

	param        t: numpy array of x-values
	param        y: numpy array of y-values
	param  alpha_l: initial lower bound on alpha (define sublevel set)
	param  alpha_u: initial upper bound on alpha
	param max_iter: max num iterations to attempt

	returns:
	      z: 5 x 1 numpy array defining coefficients of P(t) and Q(t)
	         = [a_0, a_1, a_2, b_1, b_2]'
	"""

	assert (alpha_u > alpha_l), "Check that the bounds are set correctly."

	z = cp.Variable(shape=(5, 1))

	for num_iter in range(0, max_iter):

	    # set alpha halfway between bounds
	    alpha = alpha_l + 0.5*(alpha_u - alpha_l)

	    # --- CVXPY: Pose Feasibility Problem --- #
	    t_stack = np.vstack((np.ones(y.size), t, t**2)).T
	    p_stack = cp.hstack((z[0], z[1], z[2]))
	    q_stack = cp.hstack((1, z[3], z[4]))

	    # define constraints (from definition of alpha-sublevel sets)
	    c1 = -alpha*t_stack*q_stack <= t_stack*p_stack - y[:, np.newaxis]*t_stack*q_stack
	    c2 = t_stack*p_stack - y[:, np.newaxis]*t_stack*q_stack <= alpha*t_stack*q_stack
	    c3 = t_stack*q_stack >= 0

	    problem = cp.Problem(cp.Minimize(0), [c1, c2, c3])
	    problem.solve()

	    # --- Check If Converged on Solution --- #
	    if problem.status == "optimal" and (alpha_u - alpha_l <= EPS):
	        print "\n Converged on a solution in %2d iterations.\n" % num_iter
	        break
	    
	    # --- Otherwise Update Bounds and Continue --- #
	    if problem.status == "optimal":
	        alpha_u = alpha
	    else:
	        alpha_l = alpha
	    
	if problem.status in ["infeasible", "unbounded"]:
	    print("\n Did not converge on a solution.\n")

	return z.value