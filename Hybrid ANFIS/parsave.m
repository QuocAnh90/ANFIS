function parsave(fname,data,TrainTargets,TrainOutputs,TestTargets,TestOutputs,table_err_Train,table_err_Test,take, NormalizationParams,directory, directory_code)
cd(directory)
save(fname,'data','TrainTargets','TrainOutputs','TestTargets','TestOutputs','table_err_Train','table_err_Test','take', 'NormalizationParams')
cd(directory_code)
end