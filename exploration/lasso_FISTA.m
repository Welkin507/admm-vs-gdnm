function x = lasso_FISTA(A, b, mu)

    % FISTA parameters
    max_iter = 1000000;
    tol = 1e-6;
    alpha = 1 / norm(A)^2;

    % Initialization
    x = zeros(size(A, 2), 1); % Initial x
    x_new = x;
    
    for k = 1:max_iter
        x_old = x;
        x = x_new;
        c = (k - 1) / (k + 2);
        y = x + c * (x - x_old);
        g = A' * (A * y - b);
        
        % Proximal step
        x_new = prox_l1(y - alpha * g, mu * alpha);
        
        % Check KKT residual
        eta = kkt_residual(x_new, A, b, mu);
        if mod(k, 1000) == 0
            fprintf('Iteration = %d, KKT residual = %f\n', k, eta);
        end
        if eta < tol
            x = x_new;
            break;
        end
    end
end

function x = prox_l1(v, lambda)
    % Proximal operator for L1 norm (soft thresholding)
    x = sign(v) .* max(abs(v) - lambda, 0);
end

%% Calculate eta, the KKT residual
%% $\eta_k:=\frac{\|x^k-\text{Prox}_{\mu\|\cdot\|_1}(x^k-A^*(Ax^k-b))\|}{1+\|x^k\|+\|Ax^k-b\|}.$

function eta = kkt_residual(x, A, b, mu)
    % Compute the KKT residual
    r = A * x - b;
    eta = norm(x - prox_l1(x - A' * r, mu), 2) / (1 + norm(x, 2) + norm(r, 2));
end