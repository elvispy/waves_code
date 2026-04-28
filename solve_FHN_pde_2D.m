% solves FHN PDE with RHS fhn_pde_rhs.m
% and Jacobian Dfhn_pde_rhs.m
% plots solution after each time tend for K iterations

close all;

tend = 10;
K = 100;  % total time = K*tend

par.Nx=300; % should be even
par.Ny=100; % should be even
Nx = par.Nx;
Ny = par.Ny;
dt = 0.1;

%% setup initial condition
par.a = 0.05;
par.gamma = 1;
par.m = 1;
par.ktilde = 0.01;
par.eps = 0.05;
par.delta = 0.01;

par.Lx = 500; 
Lx = par.Lx;
par.hx = Lx/(Nx-1); hx = par.hx;
x = (0:Nx-1)'*hx;

par.Ly = 500;
Ly = par.Ly;
par.hy = Ly/(Ny-1); hy = par.hy;
y = (0:Ny-1)'*hy;

load('FHN_front_newton');

u0 = repmat(Uout(1:Nx), [Ny,1]);
v0 = repmat(Uout(Nx+1:2*Nx), [Ny,1]);

sol = [u0;v0]+0.01*rand(2*Nx*Ny,1);
par.c = Uout(end);
par.k = par.ktilde*par.c;

%% differentiation matrices
e0x = ones(Nx,1);
e0y = ones(Ny,1);

% identity matrices
ex = sparse(1:Nx,[1:Nx],e0x,Nx,Nx); % Nx identity
ey = sparse(1:Ny,[1:Ny],e0y,Ny,Ny); % Ny identity

% d_x
Dx = sparse(1:Nx-1,[2:Nx-1 Nx],ones(Nx-1,1)/2,Nx,Nx);
Dx = (Dx - Dx')/hx; 
Dx(1,2) = 0; Dx(Nx,Nx-1) = 0; % Neumann boundary conditions

% d_y
Dy = sparse(1:Ny-1,[2:Ny-1 Ny],ones(Ny-1,1)/2,Ny,Ny);
Dy = (Dy - Dy')/hy; 
Dy(1,2) = 0; Dy(Ny,Ny-1) = 0; % Neumann boundary conditions

% d_xx
D2x = sparse(1:Nx-1,[2:Nx-1 Nx],ones(Nx-1,1),Nx,Nx) - sparse(1:Nx,[1:Nx],e0x,Nx,Nx);
D2x = (D2x + D2x');
%D2x(1,Nx) = 1; D2x(Nx,1) = 1; % Periodic boundary conditions
D2x(1,2)=2; D2x(Nx,Nx-1)=2; % Neumann boundary conditions
D2x = D2x/hx^2;

% d_yy
D2y = sparse(1:Ny-1,[2:Ny-1 Ny],ones(Ny-1,1),Ny,Ny) - sparse(1:Ny,[1:Ny],e0y,Ny,Ny);
D2y = (D2y + D2y');
D2y(1,2)=2; D2y(Ny,Ny-1)=2; % Neumann boundary conditions
%D2y(1,Ny) = 1; D2y(Ny,1) = 1; % Periodic boundary conditions
D2y = D2y/hy^2;

% create differentiation matrices
Dx = sparse(kron(ey,Dx));
Dy = sparse(kron(Dy,ex));
D2x = sparse(kron(ey,D2x));
D2y = sparse(kron(D2y,ex));

par.Dx = Dx;
par.Dy = Dy;
par.D2x = D2x;
par.D2y = D2y;

%% solve PDE

solution= [];
times = [];

options=odeset('RelTol',1e-8,'AbsTol',1e-8,'Jacobian',@(t,y)Dfhn_pde_rhs(t,y,par));

for j=0:K-1
    j
    time = [0:dt:tend];
    sol = ode15s(@(t,y)fhn_pde_rhs(t,y,par), time, sol,options);
    times = [times j*tend];
    sol = sol.y(:,end);
    solution = [solution sol];
end

%% save solution
save('FHN_front','solution');

%% plot solution

for i =1: length(times)
    figure (1)
    surf (x,y,reshape(solution(1:Nx*Ny,i),[Nx,Ny])')
    xlabel('$\xi = x - ct$','Interpreter', 'latex')
    ylabel('y')
    set (gca,'YDir','normal') ;
    shading flat
    drawnow
    pause;
end

function f = fhn_pde_rhs(t,y,par)

% parameters
Nx=par.Nx;
Ny=par.Ny;
a=par.a;
k=par.k;
m=par.m;
eps = par.eps;
delta=par.delta;
c=par.c;
gamma = par.gamma;

% solution
u = y(1:Nx*Ny);
v = y(Nx*Ny+1:2*Nx*Ny);

Dx = par.Dx;
Dy = par.Dy;
D2x = par.D2x;
D2y = par.D2y;

% A = par.A;
% D1 = par.D1;

%% RHS FHN

chemotaxis = (k/eps)*Dx*(u.*Dx*v) + (k/eps)*Dy*(u.*Dy*v);

f1 = c*Dx*u + (D2x+D2y)*u + m*(u-a).*u.*(1-u) - chemotaxis;
f2 = c*Dx*v + delta*D2x*v + eps*(u - gamma*v);

f = [f1; f2];
end

function Df = Dfhn_pde_rhs(t,y,par)

% parameters
Nx=par.Nx;
Ny=par.Ny;
a=par.a;
k=par.k;
m=par.m;
eps = par.eps;
delta=par.delta;
c=par.c;
gamma = par.gamma;

% solution
u = y(1:Nx*Ny);
v = y(Nx*Ny+1:2*Nx*Ny);

Dx = par.Dx;
Dy = par.Dy;
D2x = par.D2x;
D2y = par.D2y;

%% Jacobian FHN

Dx = par.Dx;
Dy = par.Dy;
D2x = par.D2x;
D2y = par.D2y;

D11 =  (D2x+D2y) + c*Dx + m*spdiags(-3*u.^2+2*(1+a).*u-a,0,Nx*Ny,Nx*Ny) - (k/eps)*Dx*spdiags(Dx*v,0,Nx*Ny,Nx*Ny) - (k/eps)*Dy*spdiags(Dy*v,0,Nx*Ny,Nx*Ny);
D12 = -(k/eps)*Dx*spdiags(u,0,Nx*Ny,Nx*Ny)*Dx - (k/eps)*Dy*spdiags(u,0,Nx*Ny,Nx*Ny)*Dy;
D21 = eps*speye(Nx*Ny,Nx*Ny);
D22 = c*Dx + delta*D2x - eps*gamma*speye(Nx*Ny,Nx*Ny);

Df=[D11,D12;D21,D22];
end