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

function i=RouletteWheelSelection(P)

    r=rand;
    
    c=cumsum(P);
    
    i=find(r<=c,1,'first');

end