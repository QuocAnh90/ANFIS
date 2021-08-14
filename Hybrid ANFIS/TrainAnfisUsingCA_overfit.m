%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPFZ104
% Project Title: Evolutionary ANFIS Traing in MATLAB: Cultural Algorithm
% Publisher: Yarpiz (www.yarpiz.com)
%
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
%
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function [bestfis results_overfit]=TrainAnfisUsingCA_overfit(fis,data,maxit)

%% Problem Definition

p0=GetFISParams(fis);

Problem.CostFunction=@(x) TrainFISCost(x,fis,data);

Problem.nVar=numel(p0);

Problem.VarMin=-1;
Problem.VarMax=1;

%% DE Params
Params.MaxIt=maxit;
Params.nPop=100;

%% Run DE
[results results_overfit]=RunCA(Problem,Params, fis, data, p0);

%% Get Results

p=results.BestSol.Position.*p0;
bestfis=SetFISParams(fis,p);

end

function [results results_overfit]=RunCA(Problem,Params, fis, data, p0)

disp('Starting CA ...');

%% Problem Definition

CostFunction=Problem.CostFunction;        % Cost Function

nVar=Problem.nVar;          % Number of Decision Variables

VarSize=[1 nVar];           % Size of Decision Variables Matrix

VarMin=Problem.VarMin;      % Lower Bound of Variables
VarMax=Problem.VarMax;      % Upper Bound of Variables

%% Harmony Search Parameters

MaxIt=Params.MaxIt;              % Maximum Number of Iterations

nPop=Params.nPop;               % Population Size

pAccept=0.35;                   % Acceptance Ratio
nAccept=round(pAccept*nPop);    % Number of Accepted Individuals

alpha=0.3;

beta=0.5;


%% Initialization

% Initialize Culture
Culture.Situational.Cost=inf;
Culture.Normative.Min=inf(VarSize);
Culture.Normative.Max=-inf(VarSize);
Culture.Normative.L=inf(VarSize);
Culture.Normative.U=inf(VarSize);

% Empty Individual Structure
empty_individual.Position=[];
empty_individual.Cost=[];

% Initialize Population Array
pop=repmat(empty_individual,nPop,1);

% Generate Initial Solutions
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position);
end

% Sort Population
[~, SortOrder]=sort([pop.Cost]);
pop=pop(SortOrder);

% Adjust Culture using Selected Population
spop=pop(1:nAccept);
Culture=AdjustCulture(Culture,spop);

% Update Best Solution Ever Found
BestSol=Culture.Situational;

% Array to Hold Best Costs
BestCost=zeros(MaxIt,1);



%% Cultural Algorithm Main Loop
RMSE_train = NaN*zeros(MaxIt, 1);
RMSE_test = NaN*zeros(MaxIt, 1);
MAE_train = NaN*zeros(MaxIt, 1);
MAE_test = NaN*zeros(MaxIt, 1);
R_train = NaN*zeros(MaxIt, 1);
R_test = NaN*zeros(MaxIt, 1);
for it=1:MaxIt
    
    % Influnce of Culture
    for i=1:nPop
        
        % % 1st Method (using only Normative component)
%         sigma=alpha*Culture.Normative.Size;
%         pop(i).Position=pop(i).Position+sigma.*randn(VarSize);
        
        % % 2nd Method (using only Situational component)
%         for j=1:nVar
%            sigma=0.1*(VarMax-VarMin);
%            dx=sigma*randn;
%            if pop(i).Position(j)<Culture.Situational.Position(j)
%                dx=abs(dx);
%            elseif pop(i).Position(j)>Culture.Situational.Position(j)
%                dx=-abs(dx);
%            end
%            pop(i).Position(j)=pop(i).Position(j)+dx;
%         end
        
        % % 3rd Method (using Normative and Situational components)
        for j=1:nVar
          sigma=alpha*Culture.Normative.Size(j);
          dx=sigma*randn;
          if pop(i).Position(j)<Culture.Situational.Position(j)
              dx=abs(dx);
          elseif pop(i).Position(j)>Culture.Situational.Position(j)
              dx=-abs(dx);
          end
          pop(i).Position(j)=pop(i).Position(j)+dx;
        end        
        
        % % 4th Method (using Size and Range of Normative component)
%         for j=1:nVar
%           sigma=alpha*Culture.Normative.Size(j);
%           dx=sigma*randn;
%           if pop(i).Position(j)<Culture.Normative.Min(j)
%               dx=abs(dx);
%           elseif pop(i).Position(j)>Culture.Normative.Max(j)
%               dx=-abs(dx);
%           else
%               dx=beta*dx;
%           end
%           pop(i).Position(j)=pop(i).Position(j)+dx;
%         end        
        
        pop(i).Cost=CostFunction(pop(i).Position);
        
    end
    
    % Sort Population
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);

    % Adjust Culture using Selected Population
    spop=pop(1:nAccept);
    Culture=AdjustCulture(Culture,spop);

    % Update Best Solution Ever Found
    BestSol=Culture.Situational;
    
    
    
    p=BestSol.Position.*p0;
    bestfis=SetFISParams(fis,p);
    
    
    file = ['results_overfit.trainedModel_', num2str(it), ' = bestfis;'];
    eval(file)
    
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
    
    
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    
end

disp('End of CA.');
disp(' ');

%% Results

results.BestSol=BestSol;
results.BestCost=BestCost;


results_overfit.RMSE_train = RMSE_train;
results_overfit.RMSE_test = RMSE_test;
results_overfit.MAE_train = MAE_train;
results_overfit.MAE_test = MAE_test;
results_overfit.R_train = R_train;
results_overfit.R_test = R_test;

results_overfit.nPop = nPop;
results_overfit.MaxIt = MaxIt;




figure;
hold on
plot(RMSE_train, '-.')
plot(RMSE_test, 'r-.')
title('RMSE-CA')

figure;
hold on
plot(MAE_train, '-.')
plot(MAE_test, 'r-.')
title('MAE-CA')

figure;
hold on
plot(R_train, '-.')
plot(R_test, 'r-.')
title('R-CA')




end
