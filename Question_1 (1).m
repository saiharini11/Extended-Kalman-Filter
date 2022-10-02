clear all;
clc;
%Extended kalman filter for non linear time varying systems
%State space model
%x(k+1)=f(x(k))+w(k)
%y(k)=h(x(k))+v(k)
%v(k) - Measurement noise, w(k) -  Process noise
%Q - Covariance of w(k) - diagonal matrix , R - Covariance of v(k)
%f(x(k)) - 2*2 , w(k)- 2*2 , h - 1*2 , v - 1*1
N=100;
%Initialization - Assume
x=zeros(2,N);
x(:,1) = [0;0];%True values
xhat(:,1)= [2;1];%Estimated values
%Assuming Q and R
Q1=[1 0;0 1];
Q2=[1 0;0 1];
%Covariance of Q(k)
Q=Q1^(1/2)*Q2^(1/2)
R=3;
p=(x(:,1)-xhat(:,1))*(x(:,1)-xhat(:,1))';
p=[p];
%Construction of jacobian matrix
syms x1 x2
f1= x1/(1+x2^2);
f2= (x1*x2)/(1+x2^2);
A11=diff(f1,x1);
A12=diff(f1,x2);
A21=diff(f2,x1);
A22=diff(f2,x2);
A=[A11 A12;A21 A22]
h=x1;
C11=diff(h,x1);
C12=diff(h,x2);
C=[C11 C12]
%Generation of true states
for k=2:N
    x(1,k)=subs(f1,{x1,x2},{x(1,k-1),x(2,k-1)});
    x(2,k)=subs(f2,{x1,x2},{x(1,k-1),x(2,k-1)});
    x(:,k)=[x(1,k);x(2,k)]+[sqrt(2)*randn;sqrt(Q(1))*randn];
end

%Generation of output
%Generate random noise v(k)
v=sqrt(R)*randn(1,N);
y=C*x+v;  
    
for k=2:N
    a11=subs(A11,{x1,x2},{xhat(1,k-1),xhat(2,k-1)});
    a12=subs(A12,{x1,x2},{xhat(1,k-1),xhat(2,k-1)});
    a21=subs(A21,{x1,x2},{xhat(1,k-1),xhat(2,k-1)});
    a22=subs(A22,{x1,x2},{xhat(1,k-1),xhat(2,k-1)});
    %a11=subs(a11,x2,x(2:k))
    a=[a11 a12;a21 a22];
    size(a);
    %Step 1 : State estimate propogation
    xhatbar1=subs(f1,{x1,x2},{xhat(1,k-1),xhat(2,k-1)});
    xhatbar2=subs(f2,{x1,x2},{xhat(1,k-1),xhat(2,k-1)});
    xhatbar=[xhatbar1;xhatbar2];
    size(xhatbar1);
    size(xhat);
    %Step 2: Error covariance propogation
    pnew=a*p*a'+Q;
    %Compute Kalman gain
    c11=subs(C11,{x1,x2},{xhat(1,k-1),xhat(2,k-1)});
    c12=subs(C12,{x1,x2},{xhat(1,k-1),xhat(2,k-1)});
    c=[c11 c12];
    K=pnew*c'/(c*pnew*c'+R);
    %Step 3: State estimate update
    yhat=subs(h,{x1,x2},{xhatbar1,xhatbar2});
    xhat(:,k)=xhatbar+K*(y(k)-yhat);
    %Step 4: Error covariance update
    p=(eye(2)-K*c)*pnew;
    
end
s=size(xhat)   
%Plotting the states
t=(1:100);
%Plotting state x1
subplot(211);
plot(t,x(1,:),'Linewidth',1);
hold on;
plot(t,xhat(1,:),'Linewidth',1.5);
legend('Actual','Estimated')
title("STATE x1")
%Plotting state x2
subplot(212);
plot(t,x(2,:),'Linewidth',1);
hold on;
plot(t,xhat(2,:),'Linewidth',1.5);
legend('Actual','Estimated')
title("STATE x2")
