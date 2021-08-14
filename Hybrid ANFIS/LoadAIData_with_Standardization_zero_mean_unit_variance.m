function data=LoadAIData_with_Standardization_zero_mean_unit_variance(Name,per,rate)
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

%% Standadization of training data: zero means and variance unit
TrainInputs_0 = TrainInputs;
%Number of observations
N=length(TrainInputs_0(:,1));
%Number of variables
M=length(TrainInputs_0(1,:));
% output array of normalised values
TrainInputs=zeros(N,M);
%Subtract mean of each Column from data
TrainInputs=TrainInputs_0-repmat(mean(TrainInputs_0),N,1);
%normalize each observation by the standard deviation of that variable
TrainInputs=TrainInputs./repmat(std(TrainInputs_0,0,1),N,1);

TrainTargets_0 = TrainTargets;
%Number of observations
N=length(TrainTargets_0(:,1));
%Number of variables
M=length(TrainTargets_0(1,:));
% output array of normalised values
TrainTargets=zeros(N,M);
%Subtract mean of each Column from data
TrainTargets=TrainTargets_0-repmat(mean(TrainTargets_0),N,1);
%normalize each observation by the standard deviation of that variable
TrainTargets=TrainTargets./repmat(std(TrainTargets_0,0,1),N,1);


%
%
%% Standadization of testing data: zero means and variance unit
% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);
TestInputs_0 = TestInputs;
TestTargets_0 = TestTargets;

%Number of observations
N=length(TestInputs(:,1));
%Number of variables
M=length(TestInputs(1,:));
%Subtract mean of each Column from data
TestInputs=TestInputs-repmat(mean(TestInputs_0),N,1);
%normalize each observation by the standard deviation of that variable
TestInputs=TestInputs./repmat(std(TestInputs_0,0,1),N,1);


%Number of observations
N=length(TestTargets(:,1));
%Number of variables
M=length(TestTargets(1,:));
%Subtract mean of each Column from data
TestTargets=TestTargets-repmat(mean(TestTargets_0),N,1);
%normalize each observation by the standard deviation of that variable
TestTargets=TestTargets./repmat(std(TestTargets_0,0,1),N,1);


%%
% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;