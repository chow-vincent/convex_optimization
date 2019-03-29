%% --- LP Solver Demo: Vincent Chow --- %%

clear all, close all, clc

% --- Problem Params --- %
m = 100;
n = 150;

% instantiate example bounded problem
rng(2, 'twister')                % seed for rand/randi/randn
A = [rand(1, n); randn(m-1, n)]; % first row pos
x0 = rand(n, 1);                 % must be positive
b = A*x0;
c = randn(n, 1);

%% Solve LP Centering Problem

% minimize c'*x - sum(log(x))
% subject to A*x == b

[x_opt, nu, r_norms, num_iter] = infeasible_newton(A, b, c, x0);
fprintf('\n --- LP Centering: Infeasible Newton Method --- \n')
fprintf('\n # Iters to Converge: %2d\n', num_iter)

figure
semilogy(1:num_iter, r_norms, '-.'), grid on
ylabel('$$||r(x,\nu)||_2$$', 'fontsize', 14, 'interpreter', 'latex')
xlabel('Iter', 'fontsize', 14, 'interpreter', 'latex')
title('Norm Residual vs. Iteration', 'fontsize', 14, 'interpreter', 'latex')

% We observe quadratic convergence of the residual.

%% Solve LP with Barrier Method

% minimize c'*x
% subject to A*x == b, x >= 0

mu_array = [1.1, 1.2, 1.8];
figure
for idx = 1:length(mu_array)
    mu = mu_array(idx);
    [x_opt, ~, history] = barrier_method(A, b, c, x0, mu);
    [xx, yy] = stairs(cumsum(history(1,:)),history(2,:));
    semilogy(xx,yy), grid on, hold on
end
ylabel('Duality Gap: $$\frac{m}{t}$$', 'fontsize', 14, 'interpreter', 'latex')
xlabel('Newton Iterations', 'fontsize', 14, 'interpreter', 'latex')
title('Barrier Method: Duality Gap vs. Newton Iterations', 'fontsize', 14, 'interpreter', 'latex')
h = legend('$$\mu=1.1$$', '$$\mu=1.2$$', '$$\mu=1.8$$');
set(h, 'location', 'NorthEast', 'interpreter', 'latex', 'fontsize', 14)

% Parameter mu affects aggressiveness of barrier method to converge on
% the optimal solution.

%% LP Solver: Feasible Test

% minimize c'*x
% subject to A*x == b, x >= 0

[~, p_opt] = lp_solver(A, b, c);

% check solution against CVX
cvx_begin quiet
    variables x(n)
    minimize(c'*x)
    subject to
        A*x == b
        x >= 0
cvx_end

fprintf('\n --- LP Solver: Feasible Test --- \n')

if abs(p_opt - cvx_optval) < 1e-3
    fprintf('\n LP Solver Opt Val : %2.2f\n', p_opt)
    fprintf('\n CVX Solver Opt Val: %2.2f\n', p_opt)
    fprintf('\n LP solver matches CVX solution.\n')
else
    fprintf('\n LP solver does not match CVX solution.\n')
end

%% LP Solver: Infeasible Test

fprintf('\n --- LP Solver: Infeasible Test --- \n')

A = [-rand(1, n); randn(m-1, n)];
b = ones(m, 1);
[~, p_opt] = lp_solver(A, b, c);

% Should correctly report problem as infeasible.


