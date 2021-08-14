function parsave_fis(fis_name, fis0, fis, trainError, stepSize, chkFis, chkEr, directory, directory_code)
cd(directory)
save(fis_name,'fis0','fis','trainError','stepSize','chkFis','chkEr')
cd(directory_code)
end