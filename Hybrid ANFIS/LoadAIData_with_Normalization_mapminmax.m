function data=LoadAIData_with_Normalization_mapminmax(Name,per,rate)
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
TrainInputs = 2*((TrainInputs_0-min(TrainInputs_0))./(max(TrainInputs_0)-min(TrainInputs_0)))-1;
TrainTargets = 2*((TrainTargets_0-min(TrainTargets_0))./(max(TrainTargets_0)-min(TrainTargets_0)))-1;
%
%
%% Normalization of testing data: min max scale [0 1]
% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);

TestInputs_0 = TestInputs;
TestTargets_0 = TestTargets;
TestInputs = 2*((TestInputs_0-min(TestInputs_0))./(max(TestInputs_0)-min(TestInputs_0)))-1;
TestTargets = 2*((TestTargets_0-min(TestTargets_0))./(max(TestTargets_0)-min(TestTargets_0)))-1;

%%
% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;