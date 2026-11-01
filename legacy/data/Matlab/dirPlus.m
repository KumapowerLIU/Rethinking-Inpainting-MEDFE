function output = dirPlus(rootPath, varargin)
%dirPlus   Recursively collect files or directories within a folder.
%   LIST = dirPlus(ROOTPATH) will search recursively through the folder
%   tree beneath ROOTPATH and collect a cell array LIST of all files it
%   finds. The list will contain the absolute paths to each file starting
%   at ROOTPATH.
%
%   LIST = dirPlus(ROOTPATH, 'PropertyName', PropertyValue, ...) will
%   modify how files and directories are selected, as well as the format of
%   LIST, based on the property/value pairs specified. Valid properties
%   that the user can set are:
%
%   GENERAL:
%     'Struct'      - A logical value determining if the output LIST should
%                     instead be a structure array of the form returned by
%                     the DIR function. If TRUE, LIST will be an N-by-1
%                     structure array instead of a cell array.
%     'Depth'       - A non-negative integer value for the maximum folder
%                     tree depth that dirPlus will search through. A value
%                     of 0 will only search in ROOTPATH, a value of 1 will
%                     search in ROOTPATH and its subfolders, etc. Default
%                     (and maximum allowable) value is the current
%                     recursion limit set on the root object (i.e.
%                     get(0, 'RecursionLimit')).
%     'ReturnDirs'  - A logical value determining if the output will be a
%                     list of files or subdirectories. If TRUE, LIST will
%                     be a cell array of subdirectory names/paths. Default
%                     is FALSE.
%     'PrependPath' - A logical value determining if the full path from
%                     ROOTPATH to the file/subdirectory is prepended to
%                     each item in LIST. The default TRUE will prepend the
%                     full path, otherwise just the file/subdirectory name
%                     is returned. This setting is ignored if the 'Struct'
%                     argument is TRUE.
%
%   FILE-SPECIFIC:
%     'FileFilter'      - A string defining a regular-expression pattern
%                         that will be applied to the file name. Only files
%                         matching the pattern will be included in LIST.
%                         Default is '' (i.e. all files are included).
%     'ValidateFileFcn' - A handle to a function that takes as input a
%                         structure of the form returned by the DIR
%                         function and returns a logical value. This
%                         function will be applied to all files found and
%                         only files that have a TRUE return value will be
%                         included in LIST. Default is [] (i.e. all files
%                         are included).
%
%   DIRECTORY-SPECIFIC:
%     'DirFilter'      - A string defining a regular-expression pattern
%                        that will be applied to the subdirectory name.
%                        Only subdirectories matching the pattern will be
%                        considered valid (i.e. included in LIST themselves
%                        or having their files included in LIST). Default
%                        is '' (i.e. all subdirectories are valid). The
%                        setting of the 'RecurseInvalid' argument
%                        determines if invalid subdirectories are still
%                        recursed down.
%     'ValidateDirFcn' - A handle to a function that takes as input a
%                        structure of the form returned by the DIR function
%                        and returns a logical value. This function will be
%                        applied to all subdirectories found and only
%                        subdirectories that have a TRUE return value will
%                        be considered valid (i.e. included in LIST
%                        themselves or having their files included in
%                        LIST). Default is [] (i.e. all subdirectories are
%                        valid). The setting of the 'RecurseInvalid'
%                        argument determines if invalid subdirectories are
%                        still recursed down.
%     'RecurseInvalid' - A logical value determining if invalid
%                        subdirectories (as identified by the 'DirFilter'
%                        and 'ValidateDirFcn' arguments) should still be
%                        recursed down. Default is FALSE (i.e the recursive
%                        searching stops at invalid subdirectories).
%
%   Examples:
%
%     1) Find all '.m' files:
%
%        fileList = dirPlus(rootPath, 'FileFilter', '\.m$');
%
%     2) Find all '.m' files, returning the list as a structure array:
%
%        fileList = dirPlus(rootPath, 'Struct', true, ...
%                                     'FileFilter', '\.m$');
%
%     3) Find all '.jpg', '.png', and '.tif' files:
%
%        fileList = dirPlus(rootPath, 'FileFilter', '\.(jpg|png|tif)$');
%
%     4) Find all '.m' files in the given folder and its subfolders:
%
%        fileList = dirPlus(rootPath, 'Depth', 1, 'FileFilter', '\.m$');
%
%     5) Find all '.m' files, returning only the file names:
%
%        fileList = dirPlus(rootPath, 'FileFilter', '\.m$', ...
%                                     'PrependPath', false);
%
%     6) Find all '.jpg' files with a size of more than 1MB:
%
%        bigFcn = @(s) (s.bytes > 1024^2);
%        fileList = dirPlus(rootPath, 'FileFilter', '\.jpg$', ...
%                                     'ValidateFcn', bigFcn);
%
%     7) Find all '.m' files contained in folders containing the string
%        'addons', recursing without restriction:
%
%        fileList = dirPlus(rootPath, 'DirFilter', 'addons', ...
%                                     'FileFilter', '\.m$', ...
%                                     'RecurseInvalid', true);
%
%   See also dir, regexp.

% Author: Ken Eaton
% Version: MATLAB R2016b - R2011a
% Last modified: 4/14/17
% Copyright 2017 by Kenneth P. Eaton
% Copyright 2017 by Stephen Larroque - backwards compatibility
%--------------------------------------------------------------------------

  % Create input parser (only have to do this once, hence the use of a
  %   persistent variable):

  persistent parser
  if isempty(parser)
    recursionLimit = get(0, 'RecursionLimit');
    parser = inputParser();
    parser.FunctionName = 'dirPlus';
    if verLessThan('matlab', '8.2')  % MATLAB R2013b = 8.2
      addPVPair = @addParamValue;
    else
      parser.PartialMatching = true;
      addPVPair = @addParameter;
    end

    % Add general parameters:

    addRequired(parser, 'rootPath', ...
                @(s) validateattributes(s, {'char'}, {'nonempty'}));
    addPVPair(parser, 'Struct', false, ...
              @(b) validateattributes(b, {'logical'}, {'scalar'}));
    addPVPair(parser, 'Depth', recursionLimit, ...
              @(s) validateattributes(s, {'numeric'}, ...
                                      {'scalar', 'nonnegative', ...
                                       'nonnan', 'integer', ...
                                       '<=', recursionLimit}));
    addPVPair(parser, 'ReturnDirs', false, ...
              @(b) validateattributes(b, {'logical'}, {'scalar'}));
    addPVPair(parser, 'PrependPath', true, ...
              @(b) validateattributes(b, {'logical'}, {'scalar'}));

    % Add file-specific parameters:

    addPVPair(parser, 'FileFilter', '', ...
              @(s) validateattributes(s, {'char'}, {'row'}));
    addPVPair(parser, 'ValidateFileFcn', [], ...
              @(f) validateattributes(f, {'function_handle'}, {'scalar'}));

    % Add directory-specific parameters:

    addPVPair(parser, 'DirFilter', '', ...
              @(s) validateattributes(s, {'char'}, {'row'}));
    addPVPair(parser, 'ValidateDirFcn', [], ...
              @(f) validateattributes(f, {'function_handle'}, {'scalar'}));
    addPVPair(parser, 'RecurseInvalid', false, ...
              @(b) validateattributes(b, {'logical'}, {'scalar'}));

  end

  % Parse input and recursively find contents:

  parse(parser, rootPath, varargin{:});
  output = dirPlus_core(parser.Results.rootPath, ...
                        rmfield(parser.Results, 'rootPath'), 0, true);
  if parser.Results.Struct
    output = vertcat(output{:});
  end

end

%~~~Begin local functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%--------------------------------------------------------------------------
% Core recursive function to find files and directories.
function output = dirPlus_core(rootPath, optionStruct, depth, isValid)

  % Backwards compatibility for fullfile:

  persistent fullfilecell
  if isempty(fullfilecell)
    if verLessThan('matlab', '8.0')  % MATLAB R2012b = 8.0
      fullfilecell = @(P, C) cellfun(@(S) fullfile(P, S), C, ...
                                     'UniformOutput', false);
    else
      fullfilecell = @fullfile;
    end
  end

  % Get current directory contents:

  rootData = dir(rootPath);
  dirIndex = [rootData.isdir];
  subDirs = {};
  validIndex = [];

  % Find valid subdirectories, only if necessary:

  if (depth < optionStruct.Depth) || optionStruct.ReturnDirs

    % Get subdirectories, not counting current or parent:

    dirData = rootData(dirIndex);
    subDirs = {dirData.name}.';
    index = ~ismember(subDirs, {'.', '..'});
    dirData = dirData(index);
    subDirs = subDirs(index);
    validIndex = true(size(subDirs));
    if any(validIndex)
      % Apply directory name filter, if specified:
      nameFilter = optionStruct.DirFilter;
      if ~isempty(nameFilter)
        validIndex = ~cellfun(@isempty, regexp(subDirs, nameFilter));
      end
      if any(validIndex)
        % Apply validation function to the directory list, if specified:
        validateFcn = optionStruct.ValidateDirFcn;
        if ~isempty(validateFcn)
          validIndex(validIndex) = arrayfun(validateFcn, ...
                                            dirData(validIndex));
        end
      end
    end
  end
  % Determine if files or subdirectories are being returned:
  if optionStruct.ReturnDirs  % Return directories
    % Use structure data or prepend full path, if specified:
    if optionStruct.Struct
      output = {dirData(validIndex)};
    elseif any(validIndex) && optionStruct.PrependPath
      output = fullfilecell(rootPath, subDirs(validIndex));
    else
      output = subDirs(validIndex);
    end
  elseif isValid  % Return files
    % Find all files in the current directory:
    fileData = rootData(~dirIndex);
    output = {fileData.name}.';

    if ~isempty(output)

      % Apply file name filter, if specified:

      fileFilter = optionStruct.FileFilter;
      if ~isempty(fileFilter)
        filterIndex = ~cellfun(@isempty, regexp(output, fileFilter));
        fileData = fileData(filterIndex);
        output = output(filterIndex);
      end

      if ~isempty(output)

        % Apply validation function to the file list, if specified:

        validateFcn = optionStruct.ValidateFileFcn;
        if ~isempty(validateFcn)
          validateIndex = arrayfun(validateFcn, fileData);
          fileData = fileData(validateIndex);
          output = output(validateIndex);
        end

        % Use structure data or prepend full path, if specified:

        if optionStruct.Struct
          output = {fileData};
        elseif ~isempty(output) && optionStruct.PrependPath
          output = fullfilecell(rootPath, output);
        end

      end

    end

  else  % Return nothing

    output = {};

  end

  % Check recursion depth:

  if (depth < optionStruct.Depth)

    % Select subdirectories to recurse down:

    if ~optionStruct.RecurseInvalid
      subDirs = subDirs(validIndex);
      validIndex = validIndex(validIndex);
    end

    % Recursively collect output from subdirectories:

    nSubDirs = numel(subDirs);
    if (nSubDirs > 0)
      subDirs = fullfilecell(rootPath, subDirs);
      output = {output; cell(nSubDirs, 1)};
      for iSub = 1:nSubDirs
        output{iSub+1} = dirPlus_core(subDirs{iSub}, optionStruct, ...
                                      depth+1, validIndex(iSub));
      end
      output = vertcat(output{:});
    end

  end

end

%~~~End local functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~