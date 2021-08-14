%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPFZ104
% Project Title: Evolutionary ANFIS Traing in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
%
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
%
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function fis=CreateInitialFIS_SubClustering_genfis2(data,Radius)


x = data.TrainInputs;
t = data.TrainTargets;

% options = genfisOptions('SubtractiveClustering');
% fis=genfis(x,t,options);

fis = genfis2(x, t, Radius);
fis.aggMethod = 'sum';
end