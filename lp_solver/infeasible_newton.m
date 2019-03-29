function [x, nu, r_norms, num_iter] = infeasible_newton(A, b, c, x0)
% INFEASIBLE_NEWTON performs the infeasible Newton step method.
% Can solve the following centering problem: 
% 
% minimize   c'*x - sum(log(x))
% subject to A*x == b
%
% param  A: m x n matrix
% param  b: m x 1 vector
% param  c: n x 1 vector of linear program
% param x0: n x 1 initial guess for Newton method (can be infeasible)
%
% returns:
%        x: n x 1 optimal point
%       nu: m x 1 optimal dual variable
%  r_norms: contains norms of the residuals ||r(x, nu)||
% num_iter: number of iterations required to converge

% --- Algorithm Params --- %
max_iter = 50; % max number of iterations to try
eps = 1e-6;    % threshold for r_norm to converge

[m, ~] = size(A);
x = x0; nu = zeros(m, 1);

% line search params
alpha = 0.1; beta = 0.4;
grad_f = c - x.^-1;
r = [(grad_f + A'*nu); A*x - b];

r_norms = []; num_iter = 0;
while norm(r, 2) > eps && num_iter < max_iter

    r_norms = [r_norms, norm(r, 2)];

    % compute step direction
    inv_hess_f = sparse(diag(x.^2));
    hess_f = sparse(diag(x.^-2));

    del_nu = (A*inv_hess_f*A')\(A*x - b - A*inv_hess_f*(grad_f + A'*nu));
    del_x = -inv_hess_f*(grad_f + A'*nu + A'*del_nu);

    % initialize line search
    t = 1;
    grad_f_new = c - (x + t*del_x).^-1;
    new_r = [grad_f_new + A'*(nu + t*del_nu); A*(x + t*del_x) - b];

    while norm(new_r, 2) > (1-alpha*t)*norm(r, 2)
        t = beta*t;
        grad_f_new = c - (x + t*del_x).^-1;
        new_r = [grad_f_new + A'*(nu + t*del_nu); A*(x + t*del_x) - b];
    end

    % update estimates
    x  = x + t*del_x; nu = nu + t*del_nu;
    grad_f = c - x.^-1;
    r = [(grad_f + A'*nu); A*x - b];

    num_iter = num_iter + 1;

end
    
end

