% - Solves FitzHugh Nagumo travelling wave equation
% - Uses finite differences and sparse matrices
% - Requires optimization toolbox

close all;
clear all;

%% setup
par.N = 300;    % number of mesh points, must be even!
par.L = 300;    % domain truncation
N = par.N;
L = par.L;
par.h = L/(N-1); h = par.h;
x = (0:N-1)'*h;
par.x = x;

%% initial conditions & parameters
par.a = 0.05;
par.gamma = 1;
par.ktilde = 0.01;
par.m = 1;
par.eps = 0.05;
a = par.a; ktilde = par.ktilde; gamma = par.gamma; eps = par.eps; m = par.m;
b = (ktilde-sqrt(ktilde^2+2*m))/2;
par.c = (m*a-b^2)/b;

% Initial Condition
u0 = [ones(N/2,1);zeros(N/2,1)];
v0 = [ones(N/2,1);zeros(N/2,1)];
U  = [u0;v0; par.c];
U0 = U;

%% Make Derivative Matrices and Linear Parts
e = ones(N,1);
z = sparse(N,N);

% d_x (central)
D = sparse(1:N-1,[2:N-1 N],ones(N-1,1)/2,N,N);
D = (D - D')/h;
D(1,2) = 0; D(N,N-1) = 0; % Neumann BCs
par.D = D;

% d_xx
D2 = sparse(1:N-1,[2:N-1 N],ones(N-1,1),N,N) - sparse(1:N,[1:N],e,N,N);
D2 = (D2 + D2');
D2(1,2)=2; D2(N,N-1)=2; % Neumann Bcs
D2 = D2/h^2;
par.D2 = D2;

%% solve nonlinear problem using fsolve

% option to display output and use Jacobian
options=optimset('Display','iter','Jacobian','on','MaxIter',10000,'Algorithm','levenberg-marquardt', 'TolFun', 1e-12);

% call solve
[Uout,fval] = fsolve(@(U) fhn_finite_differences(U,U0,par),U,options);

%% plot results

u = Uout(1:N);
v = Uout(N+1:2*N);
c = Uout(end);

figure(1); subplot(1,2,1);
plot(x,u); hold on;
title('u-profile'); xlabel('x'); ylabel('u');
subplot(1,2,2);
plot(x,v);
title('v-profile'); xlabel('x'); ylabel('v');

%% Compute critical spectral curve

lambda_crit = [];
ell = 0:0.0001:0.001;

for i=1:length(ell)
    i
    D11 = D2-ell(i)^2 + c*D + m*spdiags(-3*u.^2+2*(1+a).*u-a,0,N,N) - (ktilde*c/eps)*D*spdiags(D*v,0,N,N);
    D12 = -(ktilde*c/eps)*D*spdiags(u,0,N,N)*D + (ktilde*c/eps)*ell(i)^2*spdiags(u,0,N,N); % ask about this line
    D21 = eps*speye(N,N);
    D22 = c*D - eps*gamma*speye(N,N);

    J=[D11,D12;D21,D22];

    [V1,DD1] = eigs(J,1,1);
    evalsn = diag(DD1);

    % figure(3)
    % plot(real(evalsn),imag(evalsn),'*');

    lambda_crit = [lambda_crit DD1(1)];

end

figure(2)
ax.FontSize = 30;
plot(ell, lambda_crit); xlabel('$\ell$','Interpreter', 'latex'); ylabel('$\lambda(\ell)$','Interpreter', 'latex');

%% save computed solution

save('FHN_front_newton','Uout');


function [F,J] = fhn_finite_differences(U,U0,par)

%% parameters

N = par.N;

a = par.a;
m = par.m;
ktilde = par.ktilde;
eps = par.eps;
gamma = par.gamma;

u = U(1:N);
v = U(N+1:2*N);
u0 = U0(1:N);
v0 = U0(N+1:2*N);
c = U(end);

D = par.D;
D2 = par.D2;


%% RHS FHN
chemotaxis = -ktilde*D*(u.*(u-gamma*v));

f1 = c*D*u + D2*u + m*(u-a).*u.*(1-u)-chemotaxis;
f2 = c*D*v + eps*(u-gamma*v);
fi = (D*u)'*(u-u0);

F = [f1; f2; fi];

%% Jacobian
if nargout > 1

    D11 = D2 + c*D+m*spdiags(-3*u.^2+2*(1+a).*u-a,0,N,N) + ktilde*D*spdiags(2*u-gamma*v,0,N,N);
    D12 = ktilde*D*spdiags(-gamma*u,0,N,N);
    D21 = eps*speye(N,N);
    D22 = c*D - eps*gamma*speye(N,N);

    J=[[D11,D12;D21,D22], [D*u;D*v];...
        (D*u)'+ u'*D - u0'*D, sparse(1,N), 0];
end

end
