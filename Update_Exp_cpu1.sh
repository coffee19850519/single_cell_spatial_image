#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH -A xulab-gpu
#SBATCH --partition hpc4
#SBATCH --cpus-per-task=1  
#SBATCH --mem-per-cpu=32G 
#SBATCH --time 2:00:00    
## labels and outputs
#SBATCH --job-name=Update-Exp-%j.out
#SBATCH --output=Exp1.0-%j.out  # %j is the unique jobID
echo "### Starting at: $(date) ###"


module load miniconda3
source activate Python_R

cd /group/xulab/Su_Li/Yuzhou_sptl/Data_source
python3 ~/data/Yuzhou_Spatial/DataReshaping/Update_pos_RGB_full_2000_20201230/position_RGB_geneName_full_and_2000_1.0_1230.py



echo "### Ending at: $(date) ###"


