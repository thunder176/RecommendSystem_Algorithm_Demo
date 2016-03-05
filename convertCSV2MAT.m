function [] = convertCSV2MAT( src_name, des_name )
% convert data from .csv to .mat if needed

M = csvread(src_name);
save(des_name, M);

end
