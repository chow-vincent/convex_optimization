function [x_opt, p_opt] = lp_solver(A, b, c)
% LP_SOLVER solves the following linear program:
% 
% minimize   c'*x
% subject to A*x == b, x >= 0
%
% Implements phase I method to check problem for feasibility, and then
% in phase II, solves the linear program if feasible.
%
% param A: m x n matrix
% param b: m x 1 vector
% param c: n x 1 vector for linear program
%
% returns:
%   x_opt: n x 1 optimal point
%   p_opt: optimal value

[~, n] = size(A);

% initial guesses (can be infeasible)
x0 = ones(n, 1); % must satisfy x >= 0
t0 = 1;

% ---- Phase I: Check if Problem Feasible, Using Barrier Method --- %
c_tilde = [zeros(n, 1); 1];
ones_vec = ones(n, 1);
A_tilde = [A, -A*ones_vec];
b_tilde = b - A*ones_vec;

z0 = x0 + (t0 - 1)*ones_vec;
z0_tilde = [z0; t0];
[z_tilde, ~, ~] = barrier_method(A_tilde, b_tilde, c_tilde, z0_tilde, 1.2);

t = z_tilde(end);
if t > 1
    fprintf('\n Problem infeasible.\n')
    x_opt = NaN; p_opt = Inf;
else
    % --- Phase II: Solve Problem with Barrier Method --- %
    [x_opt, ~, ~] = barrier_method(A, b, c, x0, 1.2);
    p_opt = c'*x_opt;
end

end