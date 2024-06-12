
function [x, out] = lasso_ADMM_dual(A, b, mu)

    opts = struct();

    if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
    if ~isfield(opts, 'sigma'); opts.sigma = 1; end
    if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
    if ~isfield(opts, 'verbose'); opts.verbose = 1; end

    preprocess_time = 0;
    iter_time = 0;
    check_KKT_time = 0;

    t_preprocess = tic;
    tt = tic;
    
    x = zeros(size(A,2),1);
    out = struct();

    [m, ~] = size(A);
    sm = opts.sigma;
    y = zeros(m,1);

    f = .5*norm(A*x - b,2)^2 + mu*norm(x,1);
    fp = inf;
    out.fvec = f;
    nrmC = inf;
    

    W = eye(m) + sm * (A * A');
    R = chol(W);

    preprocess_time = toc(t_preprocess);

    for k = 1:opts.maxit
        t_iter = tic;

        fp = f;

        z = proj( - A' * y + x / sm, mu);

        h = A * (- z*sm + x) - b;
        y = R \ (R' \ h);

        c = z + A' * y;
        x = x - opts.gamma * sm * c;

        iter_time = iter_time + toc(t_iter);
        t_check_KKT = tic;

        nrmC = norm(c,2);

        f = .5*norm(A*x - b,2)^2 + mu*norm(x,1);

        k = k + 1;
        out.fvec = [out.fvec; f];

        eta = kkt_residual(x, A, b, mu);
        if opts.verbose
            fprintf('Iter: %d, Value: %f, KKT Residual: %f\n', k, f, eta);
        end

        check_KKT_time = check_KKT_time + toc(t_check_KKT);

        if eta < 1e-6
            break;
        end
    end

    out.y = y;
    out.fval = f;
    out.itr = k;
    out.tt = toc(tt);
    out.nrmC = nrmC;
    fprintf('Preprocess time: %f\n', preprocess_time);
    fprintf('Iter time: %f\n', iter_time);
    fprintf('Check KKT time: %f\n', check_KKT_time);

    end
    
    function w = proj(x, t)
    w = min(t, max(x, -t));
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