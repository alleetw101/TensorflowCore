# MRIImagesExportDS003434
# https://github.com/alleetw101/TensorflowCore <2020>
#
# fMRI/MRI representations using dataset from https://openneuro.org/datasets/ds003434/versions/1.0.1
#
import numpy as np
from PIL import Image
import MRIProcessingDS003434

data = MRIProcessingDS003434.load_mri_scan('sub-01-ses-01-anat-sub-01_ses-01_T1w.nii.gz', denoise=True, expand_dims=False, denoise_upper=0.35)
data = MRIProcessingDS003434.sagittal_slices(data)
data *= 255

for index in range(len(data)):
    image = Image.fromarray(data[index])
    image = image.convert('L')
    image.save(f'TestSagittalSliceImages/sub-01-ses-01-mask-S{index:03d}.png', transparency=0.0)

