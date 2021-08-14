%% Start
%% ANN HYBRID
clear all;close all;clc
%%
% MAIN PARAMS TO BE CHOSEN HERE:
NameData='data_su';
nCluster = 10; % number of cluster FCM
rate=0.7; % Percent of trainning
maxit = 500; % Max iteration of metaheuristic optimization

% NormalizationMethod = 1; % [range mapstd]
NormalizationMethod = 2; % [range 0, 1]
% NormalizationMethod = 3; % [range -1, 1]


% MetaHeuristicMethod = 'FA'; % Firefly
MetaHeuristicMethod = 'PSO'; % BBO
% MetaHeuristicMethod = 'GA'; % GA
% MetaHeuristicMethod = 'TLBO'; % Teaching-Learning-based Optimization
% MetaHeuristicMethod = 'PSO'; % PSO
% MetaHeuristicMethod = 'CA'; % Cultural Algorithm
% MetaHeuristicMethod = 'DE'; % Differential Evolution
% MetaHeuristicMethod = 'HS'; % Harmony Search
% MetaHeuristicMethod = 'TNSQ'; % Tabu Search for n-Queens Problem
% MetaHeuristicMethod = 'ABC'; % Artifical Bee Colony
% MetaHeuristicMethod = 'ACOR'; % TLBO
% MetaHeuristicMethod = 'SFLA'; % Shuffled Frog Leaping Algorithm
% MetaHeuristicMethod = 'ACOR'; % Ant Colony Optimization for Continuous Domains
% MetaHeuristicMethod = 'ICA'; % Imperialist Competitive Algorithm
% MetaHeuristicMethod = 'IWO'; % Invasive Weed Optimization
% MetaHeuristicMethod = 'SCE'; % Shuffled Complex Evolution

%%

Choice= [1]; % Choice of method to run

nbrun=1; % Nb of run for ANFIS and ANN: Monte Carlo


nbex=0; % nb params excluded
per=100;

for iiter=1:nbrun
    
    load(NameData);
    
    %
    Inputs_0 = Inputs;
    Targets_0 = Targets;
    %
    
    %     %     return
    %
    %     % Suffer data
    %     % ex=7;Inputs(:,ex)=0;
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
    
    data_0.TrainInputs = TrainInputs;
    data_0.TrainTargets = TrainTargets;
    data_0.TestInputs = TestInputs;
    data_0.TestTargets = TestTargets;
    
    
    [data.TrainInputs NormalizationParams_TrainInputs] = Data_Normalization_Standadization(TrainInputs, NormalizationMethod);
    [data.TrainTargets NormalizationParams_TrainTargets] = Data_Normalization_Standadization(TrainTargets, NormalizationMethod);
    data.TestInputs = Data_Normalization_Standadization_For_TestingPart(TestInputs, NormalizationMethod, NormalizationParams_TrainInputs);
    data.TestTargets = Data_Normalization_Standadization_For_TestingPart(TestTargets, NormalizationMethod, NormalizationParams_TrainTargets);
    
    
    
    nb_input=size(data.TestInputs,2);
    take=combnk(1:nb_input,nb_input-nbex);
    numsim=1;
    numsim=size(take,1);
    for icombi=1:numsim
        mat_choice=take(icombi,:);
        olddt=data;
        data.TrainInputs=data.TrainInputs(:,mat_choice);
        data.TestInputs =data.TestInputs(:,mat_choice);
        fis0=CreateInitialFIS(data,nCluster);
        %         fis0=CreateInitialFIS_SubClustering_genfis2(data,1.55);
        
        for ichoice = 1:length(Choice)
            
            %
            if Choice(ichoice)==1
                %% Choice 1 Genetic Algorithm
                if nbex==0;icombi=0;end;
                warning('off','all')
                warning
                myflag = true;
                while myflag
                    try
                        file = ['[fis results_overfit]=TrainAnfisUsing', MetaHeuristicMethod, '_overfit(fis0,data,maxit);'];
                        eval(file)
                        myflag = false;
                    end
                end
                
                
                
                
                figure;
                hold on
                plot(results_overfit.RMSE_train, '-.')
                plot(results_overfit.RMSE_test, 'r-.')
                title('Optimization procedure: RMSE')
                xlabel('Iteration')
                ylabel('Cost function RMSE')
                
                figure;
                hold on
                plot(results_overfit.MAE_train, '-.')
                plot(results_overfit.MAE_test, 'r-.')
                title('Optimization procedure: MAE')
                xlabel('Iteration')
                ylabel('Cost function MAE')
                
                figure;
                hold on
                plot(results_overfit.R_train, '-.')
                plot(results_overfit.R_test, 'r-.')
                title('Optimization procedure: R')
                xlabel('Iteration')
                ylabel('Cost function R')
                
                
                
                
                
                it_optim = maxit; %% or normaly the best iteration: to be chosen
                file = ['fis_optim = results_overfit.trainedModel_', num2str(it_optim), ';'];
                eval(file)
                
                fis = fis_optim;
                
                TrainOutputs=evalfis(data.TrainInputs,fis);TestOutputs=evalfis(data.TestInputs,fis);
                ErrorsTrain=data.TrainTargets-TrainOutputs;ErrorsTest=data.TestTargets-TestOutputs;
                % Train
                MSETrain=mean(ErrorsTrain.^2);RMSETrain=sqrt(MSETrain);MAETrain=mae(ErrorsTrain);
                error_meanTrain=mean(ErrorsTrain);error_stdTrain=std(ErrorsTrain);
                [r1,~,~] = regression(data.TrainTargets,TrainOutputs,'one');errorRTrain=r1(1);
                [~,~,~,~,stats] = regress(data.TrainTargets,TrainOutputs);errorR2Train=stats(1);
                % Test
                MSETest=mean(ErrorsTest.^2);RMSETest=sqrt(MSETest);MAETest=mae(ErrorsTest);
                error_meanTest=mean(ErrorsTest);error_stdTest=std(ErrorsTest);
                [r1,~,~] = regression(data.TestTargets,TestOutputs,'one');errorRTest=r1(1);
                [~,~,~,~,stats] = regress(data.TestTargets,TestOutputs);errorR2Test=stats(1);
                % Save to table
                table_err_Train=[RMSETrain MAETrain error_meanTrain error_stdTrain errorRTrain errorR2Train];
                table_err_Test=[RMSETest MAETest error_meanTest error_stdTest errorRTest errorR2Test];
                % Save
                %                 filename=[NameData,sprintf('%02d',Choice(ichoice)),sprintf('%03d',iiter),sprintf('%02d',icombi),'.mat'];
                TrainTargets=data.TrainTargets;TestTargets=data.TestTargets;
                %                 parsave(filename,data,TrainTargets,TrainOutputs,TestTargets,TestOutputs,table_err_Train,table_err_Test,take)
                
                figure; plotregression(TrainTargets, TrainOutputs, 'ANFIS-GA: Train Data');
                figure; plotregression(TestTargets, TestOutputs, 'ANFIS-GA: Test Data');
                
            figure;
            subplot(2,2,[1 2]);
            plot(TrainTargets,'k');
            hold on;
            plot(TrainOutputs,'r');
            legend('Target','Output');
            title('train');
            xlabel('Sample Index');
            ylabel('Actual and predicted outputs');
            grid on;
                
            subplot(2,2,3);
            plot(ErrorsTrain);
            legend('Error');
            title(['MSE = ' num2str(MSETrain) ', MAE = ' num2str(MAETrain) ', RMSE = ' num2str(RMSETrain)]);
            xlabel('Sample Index');
            ylabel('Error');
            grid on;

            subplot(2,2,4);
            histfit(ErrorsTrain, 50);
            title(['Error Mean = ' num2str(error_meanTrain) ', Error St.D. = ' num2str(error_stdTest)]);
            xlabel('Error');
            ylabel('Frequency');
            grid on;
            
            figure;
            subplot(2,2,[1 2]);
            plot(TestTargets,'k');
            hold on;
            plot(TestOutputs,'r');
            legend('Target','Output');
            title('Test');
            xlabel('Sample Index');
            ylabel('Actual and predicted outputs');
            grid on;
                
            subplot(2,2,3);
            plot(ErrorsTest);
            legend('Error');
            title(['MSE = ' num2str(MSETest) ', MAE = ' num2str(MAETest) ', RMSE = ' num2str(RMSETest)]);
            xlabel('Sample Index');
            ylabel('Error');
            grid on;

            subplot(2,2,4);
            histfit(ErrorsTest, 50);
            title(['Error Mean = ' num2str(error_meanTest) ', Error St.D. = ' num2str(error_meanTest)]);
            xlabel('Error');
            ylabel('Frequency');
            grid on;
                
            end
        end
        data=olddt;
    end
end

