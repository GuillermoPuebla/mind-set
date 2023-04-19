#!/bin/bash

folder=NAPvsMP
types=(NAPvsMP_standard NAPvsMPlines NAPvsMPnoshades NAPvsMPsilh)  ## sh doesn't like this command, use bash!
basename=(base NS base base)

for idx in ${!types[@]}
do
  exp_folder=${folder}/${types[idx]}/
  python -m src.utils.compute_distance.compute_distance_across_folders \
                              --base_folder_name ${basename[idx]} \
                              --folder ./data/${exp_folder} \
                              --result_folder ./results/${exp_folder} \
                              --affine_transf_code "tsr" \
                              --repetitions 200


  python -m src.utils.compute_distance.examples.analysis_across_folders \
                              --pickle_path ./results/${exp_folder}/cossim_df.pickle \
                              --result_folder ./results/${exp_folder}

done

