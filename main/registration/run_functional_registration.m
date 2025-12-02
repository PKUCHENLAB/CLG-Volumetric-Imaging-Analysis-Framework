function run_functional_registration(baseDir, varargin)
% RUN_FUNCTIONAL_REGISTRATION Perform 3D registration of functional calcium imaging data to structural templates.
%
% This script registers denoised functional data to a high-resolution structural template using NoRMCorre.
% It applies the calculated motion correction shifts to both the denoised data and (optionally) the raw noisy data.
%
% USAGE:
%   run_functional_registration(baseDir)
%   run_functional_registration(baseDir, 'Name', Value, ...)
%
% INPUTS:
%   baseDir - Root directory containing the experiment data subfolders.
%
% OPTIONAL PARAMETERS (Name-Value pairs):
%   'TemplateDir'   - Directory name for structural templates (default: '3dtemplate')
%   'DenoisedDir'   - Directory name for input denoised data (default: 'result_denoised')
%   'RawDataDir'    - Directory name for input raw/noisy data (default: 'result_for_denoise')
%   'OutputDir'     - Directory name for saving registered output (default: 'result_denoised_registed')
%   'RawOutputDir'  - Directory name for saving registered raw data (default: 'result_noised_registed')
%   'MappingFile'   - Name of the text file defining functional-to-structural slice mapping (default: '*.txt')
%   'ProcessRaw'    - Boolean flag to process raw data (default: true)
%   'StartSlice'    - Starting slice index (default: 2)
%   'EndSlice'      - Ending slice index (default: []) (If empty, determined from mapping file)
%
% EXAMPLE:
%   run_functional_registration('Z:\Data\Fish4\', ...
%       'TemplateDir', '3dtemplate', ...
%       'ProcessRaw', true);
%
% DEPENDENCIES:
%   - NoRMCorre (https://github.com/flatironinstitute/NoRMCorre)
%   - tiffreadVolume (Third-party utility)
%   - saveastiff (Third-party utility)
%
% CITATION:
%   If you use this code, please cite our paper:
%   [Authors]. Comprehensive Label–Guided Volumetric Imaging Enables Accurate Single-Neuron Mapping...
%

    %% 1. Parse Inputs and Configuration
    p = inputParser;
    addRequired(p, 'baseDir', @ischar);
    addParameter(p, 'TemplateDir', '3dtemplate', @ischar);
    addParameter(p, 'DenoisedDir', 'result_denoised', @ischar);
    addParameter(p, 'RawDataDir', 'result_for_denoise', @ischar);
    addParameter(p, 'OutputDir', 'result_denoised_registed', @ischar);
    addParameter(p, 'RawOutputDir', 'result_noised_registed', @ischar);
    addParameter(p, 'TempRegDir', 'template_registed', @ischar); % Output for debug templates
    addParameter(p, 'MappingFile', '*.txt', @ischar);
    addParameter(p, 'ProcessRaw', true, @islogical);
    addParameter(p, 'StartSlice', 2, @isnumeric);
    addParameter(p, 'EndSlice', [], @isnumeric);
    
    parse(p, baseDir, varargin{:});
    opts = p.Results;

    % Add NoRMCorre to path if not already present
    % Ensure this path is correct or ask user to add it manually
    if isempty(which('NoRMCorreSetParms'))
        error('NoRMCorre is not found in the MATLAB path. Please add it.');
    end

    %% 2. Load Slice Mapping
    % Find the mapping text file (e.g., ptz.txt)
    txtFiles = dir(fullfile(opts.baseDir, opts.MappingFile));
    if numel(txtFiles) ~= 1
        error('Found %d text files in %s. Expected exactly 1 mapping file matching "%s".', ...
            numel(txtFiles), opts.baseDir, opts.MappingFile);
    end
    mappingPath = fullfile(txtFiles(1).folder, txtFiles(1).name);
    fprintf('Loading slice mapping from: %s\n', mappingPath);
    
    % Read mapping file: Format "FuncSlice-StructSlice" (e.g., 1-48)
    try
        sliceTable = readtable(mappingPath, 'Delimiter', '-', 'ReadVariableNames', false);
        % Assuming 1st column is Functional Slice Index (checking consistency), 2nd is Template Slice Index
        % We construct a map where index is functional slice, value is template slice.
        % Note: The text file might not be sorted or might miss slices, so we map carefully.
        funcIndices = table2array(sliceTable(:,1));
        structIndices = table2array(sliceTable(:,2));
        
        % Create a map for easy lookup: map(functional_slice) = structural_slice
        maxFuncSlice = max(funcIndices);
        sliceMap = zeros(maxFuncSlice, 1);
        sliceMap(funcIndices) = structIndices;
    catch ME
        error('Failed to parse mapping file. Ensure format is "FuncIdx-StructIdx".\nError: %s', ME.message);
    end

    %% 3. Load Structural Template Stack
    templateFiles = dir(fullfile(opts.baseDir, opts.TemplateDir, '*.tif*'));
    if numel(templateFiles) ~= 1
        error('Found %d template files in %s. Expected exactly 1.', numel(templateFiles), fullfile(opts.baseDir, opts.TemplateDir));
    end
    templatePath = fullfile(templateFiles(1).folder, templateFiles(1).name);
    fprintf('Loading structural template from: %s\n', templatePath);
    
    try
        templateStack = tiffreadVolume(templatePath);
    catch
        error('Failed to read template TIFF using tiffreadVolume. Ensure the utility is in path.');
    end

    %% 4. Main Processing Loop
    % Determine range of slices to process
    if isempty(opts.EndSlice)
        endZ = maxFuncSlice;
    else
        endZ = opts.EndSlice;
    end
    
    % Create output directories
    if ~exist(fullfile(opts.baseDir, opts.OutputDir), 'dir'), mkdir(fullfile(opts.baseDir, opts.OutputDir)); end
    if opts.ProcessRaw && ~exist(fullfile(opts.baseDir, opts.RawOutputDir), 'dir')
        mkdir(fullfile(opts.baseDir, opts.RawOutputDir)); 
    end
    if ~exist(fullfile(opts.baseDir, opts.TempRegDir), 'dir'), mkdir(fullfile(opts.baseDir, opts.TempRegDir)); end

    fprintf('Starting registration from Slice %d to %d...\n', opts.StartSlice, endZ);

    for z = opts.StartSlice : endZ
        % Check if output already exists to skip
        outputFile = fullfile(opts.baseDir, opts.OutputDir, sprintf('z_%02d.tif', z));
        if exist(outputFile, 'file')
            fprintf('Slice z_%02d output exists. Skipping.\n', z);
            continue;
        end
        
        % Check if we have a mapping for this slice
        if z > length(sliceMap) || sliceMap(z) == 0
            warning('No structural mapping found for Functional Slice %d. Skipping.', z);
            continue;
        end
        
        structSliceIdx = sliceMap(z);
        fprintf('Processing Slice z_%02d (Template z=%d)...\n', z, structSliceIdx);
        
        % Get the specific structural template slice
        % Note: Check bounds of templateStack
        if structSliceIdx > size(templateStack, 3)
            error('Mapped structural slice %d exceeds template stack depth %d.', structSliceIdx, size(templateStack, 3));
        end
        templateImg = templateStack(:, :, structSliceIdx);
        templateImg = single(templateImg);
        templateImg = templateImg - min(templateImg(:)); % Normalize baseline
        
        %% 4.1 Load Denoised Functional Data
        denoisedFile = fullfile(opts.baseDir, opts.DenoisedDir, sprintf('denoised_z_%02d.tif', z));
        if ~exist(denoisedFile, 'file')
            warning('Denoised file not found: %s. Skipping.', denoisedFile);
            continue;
        end
        
        Y_denoised = tiffreadVolume(denoisedFile);
        [h, w, T] = size(Y_denoised);
        
        % Resize if dimensions mismatch (Functional often lower res than Structural)
        [th, tw] = size(templateImg);
        if h ~= th || w ~= tw
            % fprintf('Resizing functional data to match template...\n');
            Y_denoised = imresize3(Y_denoised, [th, tw, T]);
        end
        
        Y_denoised_single = single(Y_denoised);
        Y_reg_input = Y_denoised_single - min(Y_denoised_single(:));

        %% 4.2 First Pass: Compute Shifts using Denoised Data
        % We use NoRMCorre to find the shifts that align the functional data (Y) to the Structural Template.
        % This is a cross-modal registration (Functional -> Structural).
        
        % Parameter setup (tuned for zebrafish/mouse data as per CLG paper)
        % Note: 'bin_width' helps with long recordings.
        options_nonrigid = NoRMCorreSetParms('d1', th, 'd2', tw, ...
            'grid_size', [256, 256], 'overlap_pre', [64, 64], ...
            'mot_uf', 4, 'bin_width', 200, ...
            'max_shift', [30, 30], 'max_dev', [8, 8], ...
            'us_fac', 50, 'iter', 2, ...
            'use_parallel', true, 'boundary', 'zero');
            
        % Determine shifts
        % normcorre_batch aligns Y to a template. Here we provide the Structural Plane as template.
        tic;
        [~, shifts, template_reg, ~] = normcorre_batch(Y_reg_input, options_nonrigid, templateImg);
        procTime = toc;
        fprintf('  Shift calculation computed in %.1f seconds.\n', procTime);

        %% 4.3 Apply Shifts to Denoised Data & Save
        % Apply the computed shifts
        Y_denoised_reg = apply_shifts(Y_denoised_single, shifts, options_nonrigid);
        
        % Save Registered Denoised Data
        saveOptions.big = true; % Use BigTIFF for large files
        saveastiff(Y_denoised_reg, outputFile, saveOptions);
        
        %% 4.4 (Optional) Apply Shifts to Raw Data & Save
        if opts.ProcessRaw
            rawFile = fullfile(opts.baseDir, opts.RawDataDir, sprintf('z_%02d.tif', z));
            if exist(rawFile, 'file')
                fprintf('  Applying shifts to raw data...\n');
                Y_raw = tiffreadVolume(rawFile);
                
                % Resize if needed
                if size(Y_raw, 1) ~= th || size(Y_raw, 2) ~= tw
                    Y_raw = imresize3(Y_raw, [th, tw, size(Y_raw, 3)]);
                end
                
                Y_raw_reg = apply_shifts(single(Y_raw), shifts, options_nonrigid);
                
                rawOutputFile = fullfile(opts.baseDir, opts.RawOutputDir, sprintf('z_%02d.tif', z));
                saveastiff(Y_raw_reg, rawOutputFile, saveOptions);
            else
                warning('Raw file configured but not found: %s', rawFile);
            end
        end

        %% 4.5 Save Debug Visualization (Template vs Registered Mean)
        % We save a comparison image to verify alignment quality
        debugFile = fullfile(opts.baseDir, opts.TempRegDir, sprintf('z_%02d_compare.tif', z));
        try
            % Concatenate Structural Template and Mean Functional Image
            % Convert to same range for visualization if possible, or just keep raw values
            saveastiff(cat(3, templateImg, template_reg), debugFile);
        catch
            warning('Failed to save debug comparison file.');
        end
        
    end

    fprintf('All slices processed successfully.\n');
end

