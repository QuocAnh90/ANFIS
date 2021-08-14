function data=LoadAIData(Name,per,rate)
load(Name);

% Suffer data
% ex=7;Inputs(:,ex)=0;
nSample=size(Inputs,1)*per/100;
S=randperm(round(nSample));
Inputs=Inputs(S,:);
Targets=Targets(S,:);

% Train Data
pTrain=rate;
nTrain=round(pTrain*nSample);
TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain,:);

% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);

% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;