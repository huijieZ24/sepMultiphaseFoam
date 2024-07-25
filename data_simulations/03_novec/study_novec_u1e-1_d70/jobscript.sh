# Account for different memory requirements of different resolutions
MEMPREPROCESS=$((200*30))M
MEMMESH=$((1000*30))M

#--------------------------------------------------------------------------------------------------
# +++ Mesh creation +++
#--------------------------------------------------------------------------------------------------

BLOCKMESHJOB=$(subopenfoam -n 1 -m ${MEMMESH} -r 0:05  --printjobid -V v2212 blockMesh)
MESHJOB_BASE=$(subopenfoam -n 1 -m ${MEMMESH} -r 2:00  -w ${BLOCKMESHJOB} --printjobid -V v2212 \
			-O "-dict ./system/snappyHexMeshDict_base -overwrite" -I snappyHexMesh)

MESHJOB_SYMM=$(subopenfoam -n 1 -m ${MEMPREPROCESS} -r 1:00  -w ${MESHJOB_BASE} --printjobid -V v2212 \
			-O "-dict ./system/snappyHexMeshDict_symm -overwrite" -I snappyHexMesh)

#--------------------------------------------------------------------------------------------------
# +++ Preprocessing +++
#--------------------------------------------------------------------------------------------------

# Initialize alpha (volume fraction) field
ALPHAJOB=$(subopenfoam -n 1 -m ${MEMPREPROCESS} -r 0:05 -w ${MESHJOB_SYMM} --printjobid -V v2212 setAlphaField)

# Domain decomposition for resolutions that require more than one core 
DECOMPOSEJOB=$(subopenfoam -n 1 -m ${MEMPREPROCESS} -r 0:08 -w ${ALPHAJOB} --printjobid -V v2212 decomposePar)



#--------------------------------------------------------------------------------------------------
# +++ Solver execution +++
#--------------------------------------------------------------------------------------------------
SOLVER="interFlow"

# If interFlow is used as solver, the user installation of OpenFOAM has to be passed to the
# scheduling system
subopenfoam -n 4 \
            -N 4 \
            -r 24:00 \
            -m 1013M \
            -w ${DECOMPOSEJOB} \
            -T $HOME/OpenFOAM/OpenFOAM-v2212 \
            -j cavityWetting-oil_novec7500-air-30 -V v2212 ${SOLVER}

