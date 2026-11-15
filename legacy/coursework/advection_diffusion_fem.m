function [nodes,phi] = advection_diffusion_fem(n,u,epsilon,L,bc,source,exact_diff_flag,fem_flag)

    %% mesh and problem parameters
    % mesh size
    h = L/n;
    % mesh nodes
    nodes = linspace(0,L,n+1)';
    
    % change epsilon if "exact diffusion" is used
    if exact_diff_flag
        epsilon = epsilon + 0.5*u*h*( coth(u*h*0.5/epsilon) - 1/(u*h*0.5/epsilon) );
    end
    
    %% set up left hand side matrix
    % build system matrix: a tridiagonal matrix with each entry a linear
    % combination of "a" and "b" defined below
    % - only the equations for the interior nodes is built as the boundary
    % values are known from the boundary conditions
    a = u/(2*h); b = epsilon/(h^2);
    i = 1:n-1;
    r = repmat(i,1,3); % row
    c = [i i+1 i+2]; % column
    v = [(-a-b)*ones(1,n-1) 2*b*ones(1,n-1) (a-b)*ones(1,n-1)]; % value
    A = sparse(r,c,v,n-1,n+1);
    
    %% set up right hand side vector
    F = zeros(n-1,1);
    if fem_flag
        % numerical integration points and weights
        [x, w] = lgwt(3,0,1); x = sort(x)'; w = sort(w)';
        x = x*h; w = w*h;
        for i = 1:n-1
            % integral on left element
            xi = (i-1)*h + x;
            Ni = (xi-(i-1)*h)./h;
            fi = source(xi);
            F(i) = sum(Ni.*fi.*w)/h;
            % integral on right element
            xi = i*h + x;
            Ni = ((i+1)*h-xi)./h;
            fi = source(xi);
            F(i) = F(i) + sum(Ni.*fi.*w)/h;
        end
    else
        F(i) = source(nodes(i));
    end
    
    %% solve and plot
    phi = zeros(n+1,1);
    % set boundary conditions exactly
    phi(1) = bc(1); phi(end) = bc(2);
    phi(2:end-1) = A(:,2:end-1)\(F - A(:,[1 end])*phi([1 end]));
    
    % plot
%      figure; hold on;
%     plot(nodes,phi);
end