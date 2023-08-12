import os

def mapping_impi_to_torch():
    os.environ['LOCAL_RANK'] = os.environ['MPI_LOCALRANKID']
    os.environ['LOCAL_SIZE'] = os.environ['MPI_LOCALNRANKS']
    os.environ['RANK'] = os.environ['PMI_RANK']
    os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
    os.environ['CROSS_RANK'] = '0'
    os.environ['CROSS_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '44434'

if 'MPI_LOCALRANKID' in os.environ: # MPI launcher
    mapping_impi_to_torch()

print('===================================')
for k,v in os.environ.items():
    print(f'{k}={v}')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
