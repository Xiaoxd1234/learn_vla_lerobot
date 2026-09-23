$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

Write-Host 'Checking Conda environment: so101-lerobot'
conda run -n so101-lerobot python --version
conda run -n so101-lerobot python -c "import torch; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_version=' + str(torch.version.cuda)); print('gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"
conda run -n so101-lerobot lerobot-info
Write-Host 'Environment check passed.'
