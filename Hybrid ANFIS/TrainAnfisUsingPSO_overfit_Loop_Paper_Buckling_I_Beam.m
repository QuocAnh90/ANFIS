%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPFZ104
% Project Title: Evolutionary ANFIS Traing in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
%
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
%
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function [bestfis results_overfit]=TrainAnfisUsingPSO_overfit_Loop_Paper_Buckling_I_Beam(fis,data,maxit, npop, w, c1, c2, factor_velocity_ini)

%% Problem Definition

p0=GetFISParams(fis);

Problem.CostFunction=@(x) TrainFISCost(x,fis,data);

Problem.nVar=numel(p0);

Problem.VarMin=-1;
Problem.VarMax=1;

%% PSO Params
Params.MaxIt=maxit;
Params.nPop=npop;
Params.w=w;
Params.c1=c1;
Params.c2=c2;
Params.factor_velocity_ini=factor_velocity_ini;

%% Run PSO
[results results_overfit]=RunPSO(Problem,Params, fis, data, p0);

%% Get Results

p=results.BestSol.Position.*p0;
bestfis=SetFISParams(fis,p);

end

function [results results_overfit]=RunPSO(Problem,Params, fis, data, p0);

%     disp('Starting PSO ...');

%% Problem Definition

CostFunction=Problem.CostFunction;        % Cost Function

nVar=Problem.nVar;          % Number of Decision Variables

VarSize=[1 nVar];           % Size of Decision Variables Matrix

VarMin=Problem.VarMin;      % Lower Bound of Variables
VarMax=Problem.VarMax;      % Upper Bound of Variables

%% PSO Parameters

MaxIt=Params.MaxIt;      % Maximum Number of Iterations

nPop=Params.nPop;        % Population Size (Swarm Size)

w=Params.w;            % Inertia Weight
% Inertia Weight Damping Ratio
c1=Params.c1;           % Personal Learning Coefficient
c2=Params.c2;           % Global Learning Coefficient
factor_velocity_ini = Params.factor_velocity_ini;
% Constriction Coefficients
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;          % Inertia Weight
% wdamp=1;        % Inertia Weight Damping Ratio
% c1=chi*phi1;    % Personal Learning Coefficient
% c2=chi*phi2;    % Global Learning Coefficient

% Velocity Limits
VelMax=factor_velocity_ini*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

BestSol.Cost=inf;

for i=1:nPop
    
    % Initialize Position
    if i>1
        particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    else
        particle(i).Position=ones(VarSize);
    end
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost<BestSol.Cost
        
        BestSol=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);

%% PSO Main Loop
RMSE_train = NaN*zeros(MaxIt, 1);
RMSE_test = NaN*zeros(MaxIt, 1);
MAE_train = NaN*zeros(MaxIt, 1);
MAE_test = NaN*zeros(MaxIt, 1);
R_train = NaN*zeros(MaxIt, 1);
R_test = NaN*zeros(MaxIt, 1);
errorStD_train = NaN*zeros(MaxIt, 1);
errorStD_test = NaN*zeros(MaxIt, 1);
for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(BestSol.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<BestSol.Cost
                
                BestSol=particle(i).Best;
                
            end
            
        end
        
    end
    
    
    
    
    
    
    
    
    p=BestSol.Position.*p0;
    bestfis=SetFISParams(fis,p);
    
    if it == MaxIt
        file = ['results_overfit.trainedModel_', num2str(it), ' = bestfis;'];
        eval(file)
    end
    
    TrainOutputs=evalfis(data.TrainInputs,bestfis);
    TestOutputs=evalfis(data.TestInputs,bestfis);
    ErrorsTrain=data.TrainTargets-TrainOutputs;
    ErrorsTest=data.TestTargets-TestOutputs;
    
    MSETrain=mean(ErrorsTrain.^2);RMSETrain=sqrt(MSETrain);MAETrain=mae(ErrorsTrain);
    MSETest=mean(ErrorsTest.^2);RMSETest=sqrt(MSETest);MAETest=mae(ErrorsTest);
    [r1,~,~] = regression(data.TrainTargets,TrainOutputs,'one');errorRTrain=r1(1);
    [r1,~,~] = regression(data.TestTargets,TestOutputs,'one');errorRTest=r1(1);
    RMSE_train(it) = RMSETrain;
    RMSE_test(it) = RMSETest;
    MAE_train(it) = MAETrain;
    MAE_test(it) = MAETest;
    R_train(it) = errorRTrain;
    R_test(it) = errorRTest;
    errorStD_train(it) = std(ErrorsTrain);
    errorStD_test(it) = std(ErrorsTest);
    
    
    
    BestCost(it)=BestSol.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    %     disp(num2str(BestCost(it)))
    
    
end

%     disp('End of PSO.');
%     disp(' ');

%% Results

results.BestSol=BestSol;
results.BestCost=BestCost;



results_overfit.RMSE_train = RMSE_train;
results_overfit.RMSE_test = RMSE_test;
results_overfit.MAE_train = MAE_train;
results_overfit.MAE_test = MAE_test;
results_overfit.R_train = R_train;
results_overfit.R_test = R_test;
results_overfit.errorStD_train = errorStD_train;
results_overfit.errorStD_test = errorStD_test;

results_overfit.nPop = nPop;
results_overfit.w = w;
results_overfit.c1 = c1;
results_overfit.c2 = c2;
results_overfit.factor_velocity_ini = factor_velocity_ini;
results_overfit.MaxIt = MaxIt;


figure;
hold on
plot(RMSE_train, '-.')
plot(RMSE_test, 'r-.')
title('RMSE-PSO')

figure;
hold on
plot(MAE_train, '-.')
plot(MAE_test, 'r-.')
title('MAE-PSO')

figure;
hold on
plot(R_train, '-.')
plot(R_test, 'r-.')
title('R-PSO')

figure;
hold on
plot(errorStD_train, '-.')
plot(errorStD_test, 'r-.')
title('error-StD-PSO')

end
