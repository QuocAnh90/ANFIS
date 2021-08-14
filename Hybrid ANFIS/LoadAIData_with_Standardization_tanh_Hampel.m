function data=LoadAIData_with_Standardization_tanh_Hampel(Name,per,rate)
%% NOTE: inputs could be any distribution: robust and effiency
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

%% Standadization of training data: zero means and variance unit
TrainInputs_0 = TrainInputs;
TrainTargets_0 = TrainTargets;
TrainInputs = 0.5*(tanh(0.01*((TrainInputs_0-mean(TrainInputs_0))./std(TrainInputs_0)))+1);
TrainTargets = 0.5*(tanh(0.01*((TrainTargets_0-mean(TrainTargets_0))./std(TrainTargets_0)))+1);

%
%% Standadization of testing data: zero means and variance unit
% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);
TestInputs_0 = TestInputs;
TestTargets_0 = TestTargets;
TestInputs = 0.5*(tanh(0.01*((TestInputs-mean(TestInputs_0))./std(TestInputs_0)))+1);
TestTargets = 0.5*(tanh(0.01*((TestTargets-mean(TestTargets_0))./std(TestTargets_0)))+1);

%%
% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;