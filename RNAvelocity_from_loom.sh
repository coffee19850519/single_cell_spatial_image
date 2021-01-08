#! /bin/bash
######################### Batch Headers #########################
#SBATCH -A xulab
#SBATCH -p BioCompute,Lewis              # use the BioCompute partition Lewis,BioCompute
#SBATCH -J 1
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two days time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # number of cores (AKA tasks)
#SBATCH --mem=64G
#################################################################
module load miniconda3
source activate conda_R
python velocity.py --bamName V1_Adult_Mouse_Brain_Coronal_Section_1_possorted_genome_bam_GDOXF.loom
python velocity.py --bamName V1_Adult_Mouse_Brain_Coronal_Section_2_possorted_genome_bam_4H8X0.loom
python velocity.py --bamName V1_Adult_Mouse_Brain_possorted_genome_bam1_1K65L.loom
python velocity.py --bamName V1_Adult_Mouse_Brain_possorted_genome_bam_F90V4.loom
python velocity.py --bamName V1_Mouse_Brain_Sagittal_Anterior_possorted_genome_bam1_EC2EY.loom
python velocity.py --bamName V1_Mouse_Brain_Sagittal_Anterior_possorted_genome_bam_EHB9Z.loom
python velocity.py --bamName V1_Mouse_Brain_Sagittal_Anterior_Section_2_possorted_genome_bam1_V5MAV.loom
python velocity.py --bamName V1_Mouse_Brain_Sagittal_Anterior_Section_2_possorted_genome_bam_M8J78.loom
python velocity.py --bamName V1_Mouse_Brain_Sagittal_Posterior_possorted_genome_bam1_5TPYZ.loom
python velocity.py --bamName V1_Mouse_Brain_Sagittal_Posterior_possorted_genome_bam_4U4K8.loom
python velocity.py --bamName V1_Mouse_Brain_Sagittal_Posterior_Section_2_possorted_genome_bam1_7AZQJ.loom
python velocity.py --bamName V1_Mouse_Brain_Sagittal_Posterior_Section_2_possorted_genome_bam_RR3J3.loom
python velocity.py --bamName V1_Mouse_Kidney_possorted_genome_bam_117KC.loom
python velocity.py --bamName V1_Mouse_Kidney_possorted_genome_bam1_O5QHS.loom