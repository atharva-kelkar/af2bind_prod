#!/bin/bash

in_dir='/import/a12/users/atkelkar/struct_pred/cofolding_protein_ligands/data/posebusters/'
out_folder_name="posebusters_af2bind_pockets"

while read file_name; do

    # file_name="7C8Q_A"
    target_pdb="${in_dir}/native_structures/${file_name}.pdb"
    out_file="${in_dir}/${out_folder_name}/${file_name}_af2bind_pocket_indices.npy"

    python af2bind_script.py --target_pdb ${target_pdb} \
                            --rescale_by_max_conf \
                            --out_file ${out_file}

done < pb_files.txt