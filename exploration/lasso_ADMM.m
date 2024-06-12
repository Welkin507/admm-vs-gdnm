function [x, out] = lasso_ADMM(A, b, mu)
    opts = struct();
    if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
    if ~isfield(opts, 'sigma'); opts.sigma = 0.03; end
    if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
    if ~isfield(opts, 'verbose'); opts.verbose = 1; end
    
    tt = tic;
    x = zeros(size(A,2),1);
    out = struct();

    [m, n] = size(A);
    sm = opts.sigma;
    y = zeros(n,1);
    z = zeros(n,1);
    
    fp = inf; nrmC = inf;
    f = Func(A, b, mu, x);
    f0 = f;
    out.fvec = f0;

    AtA = A'*A;
    R = chol(AtA + opts.sigma*eye(n));
    Atb = A'*b;
    
    for k = 1:opts.maxit
        fp = f;
        
        w = Atb + sm*z - y;
        x = R \ (R' \ w);
        
        c = x + y/sm;
        z = prox(c,mu/sm);

        y = y + opts.gamma * sm * (x - z);
        f = Func(A, b, mu, x);
        nrmC = norm(x-z,2);
        
        out.fvec = [out.fvec; f];
        eta = kkt_residual(x, A, b, mu);
        fprintf('Iter: %d, Value: %f, KKT Residual: %f\n', k, f, eta);
        if eta < 1e-6
            break;
        end
    end

    out.y = y;
    out.fval = f;
    out.itr = k;
    out.tt = toc(tt);
    out.nrmC = norm(c - y, inf);
    end
    
    function y = prox(x, mu)
    y = max(abs(x) - mu, 0);
    y = sign(x) .* y;
    end

    function f = Func(A, b, mu, x)
    w = A * x - b;
    f = 0.5 * (w' * w) + mu*norm(x, 1);
    end

    %% Calculate eta, the KKT residual
    %% $\eta_k:=\frac{\|x^k-\text{Prox}_{\mu\|\cdot\|_1}(x^k-A^*(Ax^k-b))\|}{1+\|x^k\|+\|Ax^k-b\|}.$

    function x = prox_l1(v, lambda)
        % Proximal operator for L1 norm (soft thresholding)
        x = sign(v) .* max(abs(v) - lambda, 0);
    end
    
    function eta = kkt_residual(x, A, b, mu)
        % Compute the KKT residual
        r = A * x - b;
        eta = norm(x - prox_l1(x - A' * r, mu), 2) / (1 + norm(x, 2) + norm(r, 2));
    end