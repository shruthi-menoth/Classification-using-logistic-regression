clc;
clear all;
close all;

%% Question 1: Generating 2D synthetic data
% Class 0
m0 = [0 0];
diag = [2 0;0 1];
theta = 0;
sigma0 = sigma(theta,diag);
rng default;
R = mvnrnd(m0,sigma0,200);
figure
h0 = plot(R(:,1),R(:,2),'+');
hold on
x0 = get(h0,'XData');
y0 = get(h0,'YData');

%% Class 1
%Component A
ma = [-2 1];
diagA = [2 0;0 1/4];
thetaA = -3*pi/4;
sigmaA = sigma(thetaA,diagA);
rng default;
Ra = mvnrnd(ma,sigmaA,floor(200/3));
%Component B
mb = [3 2];
diagB = [3 0;0 1];
thetaB = -3*pi/4;
sigmaB = sigma(thetaB,diagB);
rng default;
Rb = mvnrnd(mb,sigmaB,ceil(200*2/3));
a = [Ra; Rb];
h1 = plot(a(:,1),a(:,2),'o');
hold on
x1 = get(h1,'XData');
y1 = get(h1,'YData');
hold off
b = [R;Ra;Rb];
hold on
x = b(:,1);
y = b(:,2);
xx = x;
yy = y;
%% Question 2: MAP Decision Boundary
db = decision_boundary(x,y,sigma0,sigmaA,sigmaB,m0,ma,mb);

syms x y;
dob = decision_boundary(x,y,sigma0,sigmaA,sigmaB,m0,ma,mb);
fimplicit(dob == 0)

%% Question 3: Conditional Probability of incorrect classification
mccount = 0;
db0 =db(1:200);
db1 =db(201:end);

for i = 1: 200
    groundtruth = 0;
    if db0(i) < 0
        db = 0;
    else
        db = 1;
    end
    if db ~= groundtruth
        mccount = mccount + 1;
    end
end
misclass0 = mccount;
misclass1=0;
for j = 1: 200
    groundtruth = 1;
    if db1(j) < 0
        db = 0;
    else
        db =1;
    end
    if db ~= groundtruth
        misclass1 = misclass1 + 1;
    end
end
misclass1
%% Question 4
% K(X,X) matrix
N = 400;
Lambda=0.5;
Q = randn([N 1]);
KernelMatrix = phi_VectorMatrix(xx,yy);
%f = featureVector(xx(1),yy(1),xx,yy)
GaussKer(1,0,1,1);
PDO = predictOutput(Q,xx,yy);
A = zeros(200,1);
B= ones(200,1);
t = cat(1,A,B);
error = PDO - t;
iterations = 10;
CF = CostFunction(PDO,t);
CFR = costFunctionReguralized(PDO,t,Q,KernelMatrix);
GCF = gradientCostFunction(KernelMatrix,error);
GCFR = gradientCostFunctionReguralized(KernelMatrix,error,Q,Lambda);
diag = Diagonal_Matrix(PDO);
hess = HessianMatrix(KernelMatrix,diag);
Rhess = ReguralizedHessianMatrix(KernelMatrix,diag,Lambda);
w = inv(hess)*GCF;
Newton = NewtonMethod(Q,xx,yy,t,iterations);
%NewtonRegularized = RegularizedNewtonMethod(Q,xx,yy,t,iterations);
%FinalPredicted = predictOutput(Newton,xx,yy);

p = linspace(-4,6,200);
q = linspace(-3,5,200);
[P,Q] = meshgrid(p,q);
%Z = sigmoidPlot2(P,Q,xx,yy,NewtonRegularized);
Z = sigmoidPlot2(P,Q,xx,yy,Newton);
contour(P,Q,Z,1);
%%
function output2 = sigmoidPlot2(X1,X2,x1,x2,a)
output = [];
for i = 1:200
    for j = 1:200
        x1Point = X1(i,j);
        x2Point = X2(i,j);
        %z = a.T.dot(featureVector(x1Point,x2Point,x1,x2))
        %z = a'.*(featureVector(x1Point,x2Point,x1,x2));
        z = dot(a',(featureVector(x1Point,x2Point,x1,x2)));
        val = sigmoid(z)-0.5;
        output = cat(1,output,val);
        
    end
end
output=reshape(output,[200,200]);
output2 = output;
end


%% Predict value
function p = predictValue(a,x1,x2,x1Pt,x2Pt)
z = a'.*(featureVector(x1Pt,x2Pt,x1,x2));
p = sigmoid(z);
end
%% Predict output

function po = predictOutput(a,x1,x2)
outputVector = [];
N = 400;
A = outputVector;
for i = 1: N
    z = dot(a',(featureVector(x1(i),x2(i),x1,x2)));
    %     s = size(featureVector)
    B = sigmoid(z);
    A = cat(1,A,B);
    po=A;
end
end

%% Decision Boundary
function db = decision_boundary(x,y,sigma0,sigmaA,sigmaB,m0,ma,mb)

K = (2*pi*sqrt(det(sigma0)));
L = 1/(2*pi*sqrt(det(sigmaA)));
M = 1/(2*pi*sqrt(det(sigmaB)));
%syms x y;
sigma0 = inv(sigma0);
sigmaA = inv(sigmaA);
sigmaB = inv(sigmaB);
% pdf1-pdf0
db = log(K)+(0.5*(sigma0(1,1).*(x-m0(1)).^2 + (sigma0(2,1)+sigma0(1,2)).*(x-m0(1)).*(y-m0(2)) + sigma0(2,2).*(y-m0(2)).^2) + ...
    + log((1/3)*L*exp(-0.5*(sigmaA(1,1).*(x-ma(1)).^2 + (sigmaA(2,1)+ sigmaA(1,2)).*(x-ma(1)).*(y-ma(2)) + sigmaA(2,2).*(y-ma(2)).^2)) + ...
    ((2/3)*M*exp(-0.5*(sigmaB(1,1).*(x-mb(1)).^2 + (sigmaB(2,1)+sigmaB(1,2)).*(x-mb(1)).*(y-mb(2)) + sigmaB(2,2).*(y-mb(2)).^2)))));
end

%% Covariance
function C = sigma(theta,diag)
V = [cos(theta) -sin(theta);sin(theta) cos(theta)];
C = V*diag*inv(V);

end
%%  Kernel
function ker = GaussKer(x0,y0,x1,y1)
l =0.2;
ker = exp(-((x0-x1)^2 + (y0-y1)^2)/(2*l^2));
end

%%  Sigmoid Function
function u = sigmoid(z)
u = 1/(1+exp(-z));
% u =inline('1.0 ./ (1.0 + exp(-z))');
end
%% Feature Vector
function f = featureVector(a,b,x,y)
featureVector1 = [];
A = featureVector1;
for i = 1:400
    
    B = GaussKer(x(i),y(i),a,b);
    A= cat(1,A,B);
end
f=A;
end
%% Feature vector matrix
function K = phi_VectorMatrix(x,y)
N = 400;
K = zeros(N);
for i = 1:N
    for j = 1:i
        ker = GaussKer(x(i),y(i),x(j),y(j));
        K(i,j) = ker;
        K(j,i) = ker;
    end
end
end

%%  Cost Function
function Ecost = CostFunction(y,t)
N = 400;
Ecost = 0;
for i= 1:N
    Ecost = Ecost - (t(i)*log(y(i))+(1-t(i))*log(1-y(i)));
end
end
%% Regularized cost function
function REcost = costFunctionReguralized(y,t,a,KernelMatrix)
REcost = 0;
Lambda=0.5;
for i=1:400
    REcost = REcost - (t(i)*log(y(i))+(1-t(i))*log(1-y(i)));
    REcost = REcost + 0.5* Lambda *(a'*KernelMatrix*a);
end
end
%% Gradient cost function
function gcf = gradientCostFunction(matrix,error)
gcf = matrix*(error);
end
%% Regularized Gradient cost function
function REgcf = gradientCostFunctionReguralized(matrix,error,a,Lambda)
REgcf = matrix*(error + Lambda*a);
end
%% Diagonal Matrix
function DiagM = Diagonal_Matrix(y)
N = 400;
DiagM = zeros(N);
for i =1:N
    DiagM(i,i) = y(i)*(1-y(i));
end
end
%% Hessian Matrix
function hm = HessianMatrix(K,R)
hm = K*R*K;
end
%% Regularized Hessian Matrix
function REhm = ReguralizedHessianMatrix(K,R,Lambda)
REhm = K*R*K + Lambda*K;
end
%% Newton Method
function Newton = NewtonMethod(a,x1,x2,t,iterations)
epsilonCost = 0.001;
prevCost= 0;
K = phi_VectorMatrix(x1,x2);
Newton = a;
for j = 1:iterations
    y = predictOutput(a,x1,x2);
    outputError = y - t;
    currCost = CostFunction(y,t);
    disp("Curr Cost")
    disp(currCost)
    disp("Previous cost")
    disp(prevCost)
    if(abs(currCost-prevCost) < epsilonCost)
        break
    end
    prevCost = currCost;
    disp("Break not executed")
    gradientCost = gradientCostFunction(K,outputError);
    R = Diagonal_Matrix(y);
    H = HessianMatrix(K,R);
    change = inv(H)*gradientCost;
    a = a - change;
    Newton = a;
    %    end
end
end
%% Newton Method Regularized
function Newton = RegularizedNewtonMethod(a,x1,x2,t,iterations)
epsilonCost = 0.001;
prevCost= 0;
K = phi_VectorMatrix(x1,x2);
for j = 1:iterations
    y = predictOutput(a,x1,x2);
    outputError = y - t;
    currCost = costFunctionReguralized(y,t,a,K);
    if(abs(currCost-prevCost) < epsilonCost)
        break
        prevCost = currCost;
        gradientCost = gradientCostFunctionReguralized(K,outputError,a);
        R = Diagonal_Matrix(y);
        H = ReguralizedHessianMatrix(K,R);
        change = inv(H)*gradientCost';
        a = a - change;
    end
end
Newton = a;
end
%% Plotting Sigmoid
function sp = sigmoidPlot(X1,X2,x1,x2,a)
output = [];
for i = 1:200
    for j = 1:200
        x1Point = X1(i,j);
        x2Point = X2(i,j);
        z = a'*dot(featureVector(x1Point,x2Point,x1,x2))
        horzcat(output,(sigmoid(z)-0.5))
    end
end
end


