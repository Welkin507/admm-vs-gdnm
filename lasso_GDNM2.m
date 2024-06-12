function x=lasso_GDNM2(A,b,mu)
    %% Find gamma 
    ATA=A'*A;
    gamma=0.5/eigs(ATA,1);
    %% Calculate Prox(y)
    function [value,zeros_index]=prox_of_function(gamma,y,mu)
        value=[];
        zeros_index=[];
        for i=1:length(y)
            if y(i)>mu*gamma
                value=[value;y(i)-mu*gamma];
            elseif y(i)<-mu*gamma
                value=[value;y(i)+mu*gamma];
            else 
                value=[value;0];
                zeros_index=[zeros_index;i];
            end
        end
    end   
    %% Calculate Q, P
    Q=inv(eye(size(ATA))-gamma*ATA);
    P=Q-eye(size(Q));
    %% In some cases, we need to use Tikonov Regularization
    %P=P+eye(size(P))*10^(-10);
    %Q=Q+eye(size(Q))*10^(-10);
    %% Calulate c
    c=-gamma*Q*transpose(A)*b;
    %% Calculate value of Lasso function
    function value=value_of_function(A,b,mu,x)
        value=0.5*norm(A*x-b)^2+mu*norm(x,1);
    end
    %% Calculate value of phi
    function value_of_phi=phi(A,b,gamma,mu,y)
        [prox_y,zeros_index]=prox_of_function(gamma,y,mu);
        value_of_phi=0.5*transpose(P*y)*y+transpose(c)*y+gamma*mu*norm(prox_y,1)+0.5*norm(y-prox_y)^2;
    end
    %% Find gradient at y
    function grad=finding_gradient(A,b,gamma,mu,y)
        [prox_y,zeros_index]=prox_of_function(gamma,y,mu);
        grad=Q*y-prox_y+c;
    end
    %% Find dk, tk at y
    function dk=finding_dk(A,b,gamma,mu,y)
        grad=finding_gradient(A,b,gamma,mu,y);
        [prox_y,zeros_index]=prox_of_function(gamma,y,mu);
        X=P;
        for i=zeros_index
            X(i,:)=Q(i,:);
        end
        dk=-linsolve(X,grad);
    end
    function tau=finding_tk(A,b,gamma,mu,y)
        tau=1;
        dk=finding_dk(A,b,gamma,mu,y);
        grad=finding_gradient(A,b,gamma,mu,y);
        while phi(A,b,gamma,mu,y+tau*dk)-phi(A,b,gamma,mu,y)-0.1*tau*grad'*dk>0
            tau=tau/2;
        end
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


    %% Start iterating
    y=zeros(length(c),1);
    iter=0;
    %while norm(finding_gradient(A,b,gamma,mu,y))>1e-10
    while iter<100
        y=y+finding_tk(A,b,gamma,mu,y)*finding_dk(A,b,gamma,mu,y);
        iter=iter+1;
        value=0.5*norm(A*(Q*y+c)-b)^2+mu*norm(Q*y+c,1);
        
        % Calculate the KKT residual
        x = Q * y + c;
        eta = kkt_residual(x, A, b, mu);
        fprintf('Iter: %d, Value: %f, KKT Residual: %f\n', iter, value, eta);
        if eta < 1e-6
            break;
        end
    end
    %% Translate y to x
    x=Q*y+c;
end
