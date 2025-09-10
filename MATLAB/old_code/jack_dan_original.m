clear all
close all

% please note that all the code is run in dimensionless coordinates
N=300;
zeta=-1/2;
theta=1;
num=10;
Omegas=logspace(-1,1,num);
xds=logspace(log10(10),log10(0.7),num); % (stretch the domain to avoid shallow water effects as we change Omega)
zds=logspace(log10(2.5),log10(0.25),num);
it=0;
for Omega=Omegas
    it=it+1;
    it
    xd=xds(it);
    zd=zds(it);
    
    fk=@(k) k.*tanh(k*zd)-Omega^2;
    kstar=fsolve(fk,Omega^2,optimset('display','off'));
    
    X=repmat(linspace(-xd,xd,N),N,1);
    Z=repmat(linspace(-zd,0,N),N,1)';
        
    zs=Z(:,end);
    xs=X(end,:);
    
    i1=find(abs(xs)<=1/2,1);
    i2=find(abs(xs)<=1/2,1,'last');
    
    dx=xs(2)-xs(1);
    dz=zs(2)-zs(1);
    
    [Dx, Dz] = getNonCompactFDmatrix2D(N,N,dx,dz,1,2);
    [Dxx, Dzz] = getNonCompactFDmatrix2D(N,N,dx,dz,2,2);
    Dx1 = getNonCompactFDmatrix(N,dx,1,2);
    
    [L0,BND]=build_matrices(N,i1,i2,Omega,kstar,dz,dx,0,0);
    BNDi=find(BND);
    
    Lap=Dxx+Dzz;
    
    [L,F] = update_matrices(Lap,xs,zs,N,BNDi,L0,theta,zeta,dz);

    y=L\F;
    
    Phi=reshape(y,N,N);
   
    phi=y;
    
    u=Dx*phi;
    w=Dz*phi;
    
    phi0=(Phi(Z==0));
    
    U=reshape(u,N,N);
    W=reshape(w,N,N);
    
    eta=(-1i*W(Z==0));
    etax=Dx1*eta;
    
    uR=(U(X==xd));
    uL=(U(X==-xd));
    
    % force integral
    int= 1/2*(real(eta).*real(etax)+imag(eta).*imag(etax)+real(Omega^2*1i*phi0).*real(etax)+imag(Omega^2*1i*phi0).*imag(etax));
    int(abs(xs')>1/2)=0;
    Fs(it)=trapz(xs',int);
    
    % momentum integral
    Mom(it)=-Omega^2/2*(trapz(zs,real(uR).^2+imag(uR).^2-real(uL).^2-imag(uL).^2));
    
    % LH calculations
    dAsqd(it)=abs(eta(1))^2-abs(eta(end))^2; 
    %dAsqd(it)=max(abs(eta(1:150)))^2-max(abs(eta(151:end)))^2;  

    ks(it)=kstar;
    
end
LH = 0.5*dAsqd.*Omegas.^2./ks;
fig4=figure(4);clf;
loglog(Omegas,Fs,'k','linewidth',3,'markersize',10); 
hold on
plot(Omegas,Mom,'r--','linewidth',3,'markersize',10);
plot(Omegas,LH,'b','linewidth',2,'markersize',10);

set(gca,'TickLabelInterpreter','latex')
set(gca,'fontsize',25)
grid on
xlabel('$$\omega\sqrt{L/g}$$','interpreter','latex')
ylabel('$\overline{F}_T/\rho g L^2 $','interpreter','latex')
legend({'$$ \overline{F}_T$$',...
    '$$[\int\overline{\rho u^2 + p}\,\mathrm{d}z]^{x=-\ell}_{x=\ell}$$',...
    'L-H scaling'},...
    'interpreter','latex','location','southeast')
box on
set(gcf,'color','w')
ylim([1e-6,1e1])
yticks(logspace(-6,2,5))
xlim([1e-1,1e1])
xticks(logspace(-1,1,3))
