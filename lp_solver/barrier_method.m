function [x, nu, history] = barrier_method(A, b, c, x0, mu)
% BARRIER_METHOD performs barrier method, using infeasible Newton step.
% Can be used in a phase I method to check for feasibility, then in 
% phase II to solve the following problem:
% 
% minimize   c'*x
% subject to A*x == b, x >= 0
%
% param  A: m x n matrix
% param  b: m x 1 vector
% param  c: n x 1 vector for linear program
% param x0: n x 1 initial guess for Newton method (can be infeasible)
%
% returns:
%       x: n x 1 optimal point
%      nu: m x 1 optimal dual variable
% history: reports duality gap and # Newton steps per centering step

m = length(b); % number of constraints
eps = 1e-3;    % threshold for duality gap
t = 0.1;       % t > 0 param

x = x0;

history = [];
while (m/t) >= eps    
    [x, nu, ~, num_iter] = infeasible_newton(A, b, t*c, x);
    t = mu*t;
    history = [history, [num_iter; (m/t)]];
end

end