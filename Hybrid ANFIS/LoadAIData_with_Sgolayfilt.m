function data=LoadAIData_with_Sgolayfilt(Name,per,rate)
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

%% 
order_poly = 2;
nb_point_filter = 31;

TrainInputs_2 = NaN*zeros(size(TrainInputs));
for i = 1:size(TrainInputs, 2)
    TrainInputs_2(:, i) = sgolayfilt(TrainInputs(:, i), order_poly, nb_point_filter);
end
TrainInputs = TrainInputs_2;
TrainTargets = sgolayfilt(TrainTargets, order_poly, nb_point_filter);

% figure;
% plot(Targets(1:nTrain,:))
% hold on
% plot(TrainTargets, 'r')

%
%
%% Normalization of testing data: min max scale [0 1]
% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);

TestInputs_2 = NaN*zeros(size(TestInputs));
for i = 1:size(TestInputs_2, 2)
    TestInputs_2(:, i) = sgolayfilt(TestInputs(:, i), order_poly, nb_point_filter);
end
TestInputs = TestInputs_2;
TestTargets = sgolayfilt(TestTargets, order_poly, nb_point_filter);

%%
% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;