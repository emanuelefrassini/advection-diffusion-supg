clear all;
L = 4;
u = 1;
n = 20;
f = 0;
h = L/n;

epsilon = 0.01;
epshat=0.5*u*h*( coth(u*h*0.5/epsilon) - 1/(u*h*0.5/epsilon) );

c0 = u/(2*h)-epsilon/(h^2);
c1 = -epsilon/(h^2);
c2 = -u/(2*h)-epsilon/(h^2);
C0 = c0*ones(1,n-1);
C00 = [0,C0];
C1 = [-1/2,c1*ones(1,n-1),-1/2];
C2 = [c2*ones(1,n-1),0];
A = h*(diag(C00,1) +  diag(-2*C1,0) +  diag(C2,-1));
A(1,1) = 1;
A(end,end) = 1;

epsi=epsilon+epshat;
c0supg = u/(2*h)-epsi/(h^2);
c1supg = -epsi/(h^2);
c2supg = -u/(2*h)-epsi/(h^2);
C0supg = c0supg*ones(1,n-1);
C00supg = [0,C0supg];
C1supg = [-1/2,c1supg*ones(1,n-1),-1/2];
C2supg = [c2supg*ones(1,n-1),0];
Asupg = h*(diag(C00supg,1) +  diag(-2*C1supg,0) +  diag(C2supg,-1));
Asupg(1,1) = 1;
Asupg(end,end) = 1;
xx = 0:h:L;
source=@(x) zeros(size(x)); %(x<=1.5).*(1-x) + (x>1.5 & x<=2).*(x-2);
for i=2:n
Fgal(i)=(h)*source(xx(i));
end
for i=2:n
Fsupg(i)=(h-epshat/u)*source(xx(i))+epshat/u*source(xx(i-1));
end

for i=2:n
Fgal(i)=(h)*source(xx(i));
end
Fgal(n + 1) = 1;
Fgal(1) = 0;
Fsupg(n + 1) = 1;
Fsupg(1) = 0;
phigal = A\Fgal';
phisupg=Asupg\Fsupg';

n=1000;
h=L/n;
xxx=0:h:L;
c0 = u/(2*h)-epsilon/(h^2);
c1 = -epsilon/(h^2);
c2 = -u/(2*h)-epsilon/(h^2);
C0 = c0*ones(1,n-1);
C00 = [0,C0];
C1 = [-1/2,c1*ones(1,n-1),-1/2];
C2 = [c2*ones(1,n-1),0];
A = h*(diag(C00,1) +  diag(-2*C1,0) +  diag(C2,-1));
A(1,1) = 1;
A(end,end) = 1;


for i=2:n
Fgal(i)=(h)*source(xxx(i));
end
Fgal(1)=0;
Fgal(n+1)=1;
phiex=A\Fgal';
% for i = 1:6  
%     %F(i) = (1/2)*(subs(xx(i),frhs) - subs(xx(i + 1) + h,frhs)-xx(i)*(subs(xx(i),f) + subs(xx(i + 1),f)) +  (xx(i + 1) + h)*(subs((xx(i + 1) + h),f) +  subs(xx(i + 1),f)));
%     F(i) = h * (1 - xx(i)); %h*subs(xx(i),frhs);
% end
% for i = 7:8 
%     F(i) = h * (xx(i) - 2); %h*subs(xx(i),frhs);
% end
% for i = 9:15 
%     F(i) = 0; %h*subs(xx(i),frhs);
% end
figure(1);
plot(xx,phigal,xx,phisupg,xxx,phiex )
 legend('Solution with Galerkin','Solution with SUPG','"Exact" solution');


