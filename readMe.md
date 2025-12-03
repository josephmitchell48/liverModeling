### Important Commands

python3 -m venv .venv
source .venv/bin/activate
deactivate

python make_hollow_liver.py \
volumeFiles/case010/hepaticvessel_010_liver.nii.gz \
volumeFiles/case010/hepaticvessel_010_vsnet.nii.gz \
--out volumeFiles/case010/hollow_liver_010.stl \
--vessel-dilation-mm 1.0
