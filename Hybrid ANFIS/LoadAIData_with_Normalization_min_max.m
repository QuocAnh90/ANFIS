function data=LoadAIData_with_Normalization_min_max(Name,per,rate)
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

%% Normalization of training data: min max scale [0 1]
TrainInputs_0 = TrainInputs;
TrainTargets_0 = TrainTargets;
TrainInputs = (TrainInputs_0-min(TrainInputs_0))./(max(TrainInputs_0)-min(TrainInputs_0));
TrainTargets = (TrainTargets_0-min(TrainTargets_0))./(max(TrainTargets_0)-min(TrainTargets_0));
%
%
%% Normalization of testing data: min max scale [0 1]
% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);

TestInputs_0 = TestInputs;
TestTargets_0 = TestTargets;
TestInputs = (TestInputs_0-min(TestInputs_0))./(max(TestInputs_0)-min(TestInputs_0));
TestTargets = (TestTargets_0-min(TestTargets_0))./(max(TestTargets_0)-min(TestTargets_0));

%%
% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;