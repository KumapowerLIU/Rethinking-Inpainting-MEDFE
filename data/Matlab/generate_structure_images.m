dataset_path='I:\places365standard_easyformat\places365_standard\val2\butte'
output_path='I:\ECCV DATA\butteval'
    image_list = dirPlus(dataset_path, 'FileFilter', '\.(jpg|png|tif)$');
    num_image = numel(image_list);
    for i=1:num_image
       image_name = image_list{i};
       image = im2double(imread(image_name));
       S = tsmooth(image, 0.015, 3, 0.001, 3);
       write_name = strrep(image_name, dataset_path, output_path);
       [filepath,~,~] = fileparts(write_name);
       if ~exist(filepath, 'dir')
           mkdir(filepath);
       end
       imwrite(S, write_name);      
       
       if mod(i,100)==0
          fprintf('total: %d; output: %d; completed: %f%% \n',num_image, i, (i/num_image)*100) ;
       end
    end

