function [x, out] = lasso_ADMM_ultra(A, b, mu)
    opts = struct();
    if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
    if ~isfield(opts, 'sigma'); opts.sigma = 0.03; end
    if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
    if ~isfield(opts, 'verbose'); opts.verbose = 1; end
    
    preprocess_time = 0;
    iter_time = 0;
    check_KKT_time = 0;
    x_time = 0;

    t_preprocess = tic;
    tt = tic;

    out = struct();

    [m, n] = size(A);
    if (n <= 256)
        opts.sigma = 0.5;
    end
    
    sm = opts.sigma;
    y = zeros(n,1);
    z = zeros(n,1);
    
    x = zeros(n,1);
    r = - b;
    f = Func(x, r, mu);
    f0 = f;
    out.fvec = f0;

    AtA = A'*A;
    R = chol(AtA + opts.sigma*eye(n));
    Atb = A'*b;
    
    preprocess_time = toc(t_preprocess);

    for k = 1:opts.maxit
        t_iter = tic;
        
        w = Atb + sm*z - y;
        t_x = tic;
        x = R \ (R' \ w);
        x_time = x_time + toc(t_x);
        
        c = x + y/sm;
        z = prox(c,mu/sm);

        y = y + opts.gamma * sm * (x - z);

        r = A * x - b;
        f = Func(x, r, mu);
        out.fvec = [out.fvec; f];

        iter_time = iter_time + toc(t_iter);
        t_check_KKT = tic;

        eta = kkt_residual(x, r, A, b, mu);
        fprintf('Iter: %d, Value: %f, KKT Residual: %f\n', k, f, eta);
        check_KKT_time = check_KKT_time + toc(t_check_KKT);

        if eta < 1e-6
            break;
        end
    end

    if opts.verbose
        fprintf('Preprocess Time: %f\n', preprocess_time);
        fprintf('Iter Time: %f\n', iter_time);
        fprintf('Check KKT Time: %f\n', check_KKT_time);
        fprintf('X Time: %f\n', x_time);
    end
end
    
    function y = prox(x, mu)
    y = max(abs(x) - mu, 0);
    y = sign(x) .* y;
    end

    function f = Func(x, r, mu)
        f = 0.5 * (r' * r) + mu*norm(x, 1);
    end

    %% Calculate eta, the KKT residual
    %% $\eta_k:=\frac{\|x^k-\text{Prox}_{\mu\|\cdot\|_1}(x^k-A^*(Ax^k-b))\|}{1+\|x^k\|+\|Ax^k-b\|}.$

    function x = prox_l1(v, lambda)
        % Proximal operator for L1 norm (soft thresholding)
        x = sign(v) .* max(abs(v) - lambda, 0);
    end
    
    function eta = kkt_residual(x, r, A, b, mu)
        % Compute the KKT residual
        eta = norm(x - prox_l1(x - A' * r, mu), 2) / (1 + norm(x, 2) + norm(r, 2));
    end