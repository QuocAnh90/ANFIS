function [data NormalizationParams] = LoadAIData_with_Normalization_Standadization(Name,per,rate, NormalizationMethod)
%% NOTE: inputs must be Gaussian distributed
%%
load(Name);
%%
% Suffer data
% ex=7;Inputs(:,ex)=0;
nSample=size(Inputs,1)*per/100;
S=randperm(round(nSample));
Inputs=Inputs(S,:);
Targets=Targets(S,:);

%%
% Train Data
pTrain=rate;
nTrain=round(pTrain*nSample);

TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain,:);

%% Normalization of training data: Data_Normalization_Sigmoid
[TrainInputs NormalizationParams_TrainInputs] = Data_Normalization_Standadization(TrainInputs, NormalizationMethod);
[TrainTargets NormalizationParams_TrainTargets] = Data_Normalization_Standadization(TrainTargets, NormalizationMethod);



%
%% Normalization of testing data: min max scale [0 1]
% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);

[TestInputs NormalizationParams_TestInputs] = Data_Normalization_Standadization(TestInputs, NormalizationMethod);
[TestTargets NormalizationParams_TestTargets] = Data_Normalization_Standadization(TestTargets, NormalizationMethod);

NormalizationParams.NormalizationParams_TrainInputs = NormalizationParams_TrainInputs;
NormalizationParams.NormalizationParams_TrainTargets = NormalizationParams_TrainTargets;
NormalizationParams.NormalizationParams_TestInputs = NormalizationParams_TestInputs;
NormalizationParams.NormalizationParams_TestTargets = NormalizationParams_TestTargets;

%%
% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;