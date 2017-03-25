function convertImages(imgPath, imgType)
    %imgPath = 'path/to/images/folder/';
    %imgType = '*.png'; % change based on image type
    images  = dir([imgPath imgType]);
    N = length(images);

    % check images
    if( ~exist(imgPath, 'dir') || N < 1 )
        display('Directory not found or no matching images found.');
    end

    % preallocate cell
    Seq{N,1} = [];

    for idx = 1:N
        [name_img, ext] = strtok(images(idx).name, '.');
        if strcmp(ext,'.jpg') == 0
            Seq{idx} = imread([imgPath images(idx).name]);
%             figure, imshow(Seq{idx});
            imgPath_ = strcat(imgPath, 'jpg/');
            newName = strcat(imgPath_, name_img);
            newName = strcat(newName, '.jpg');
            imwrite(Seq{idx}, newName);
        else
            imgPath_ = strcat(imgPath, 'jpg/');
            newName = strcat(imgPath_, images(idx).name);
            imwrite(Seq{idx}, newName);
        end
    end
end