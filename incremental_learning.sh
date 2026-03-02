#!/bin/bash
mkdir -p logs








fine_tune=0 #the weight to force the fine-tuning to align domains
#intrinsic_ell=0.15 #nw a bool
#/ np.mean(ngal[ngal!=0]) = 0.147
adaptation_weight=1


function get_zs () {
    
    if [[ $1 == "concat" ]]; then
        zs=1.65
    elif [[ $1 == "f115w" ]]; then
        zs=1.6
    elif [[ $1 == "f150w" ]]; then
        zs=1.65
    else
        echo "NOT FOUND"
        exit
    fi
    
    echo $zs
}

for FILTER in concat
do
    ZS=$(get_zs "${FILTER}")

    for SEED in {3..30}
    do
        
        SRC="bahamas"
        TGT="darkskies"
        
        
        TGT_LAB=`echo $TGT | cut -c1-4`
        SRC_LAB=`echo $SRC | cut -c1-4`

        COMMON_ARGS="--batch_size 32 --epochs 100 --lr 0.0001 --eval_interval 10 --project_name shear_input --image_size 100 --use_wandb"
        COMMON_ARGS=$COMMON_ARGS" --model squeezenet1_1 --pretrained --mass_index 0 --in_channels 2 --verbose --weighting_scheme inverse_frequency"


        RUN_NAME=cdan_${SRC_LAB}2${TGT_LAB}_pre_squeezenet1_aw_${adaptation_weight}_pad_shear_avgpool_gauss_seed_${SEED}_nob1

        BASE_DIR=models/base_models/


        if [ ! -f ${BASE_DIR}/${RUN_NAME}_final.pth ]
        then


        python main.py $COMMON_ARGS \
               --source_domain $SRC --target_domain $TGT \
                --run_name ${RUN_NAME} \
                --adaptation_weight ${adaptation_weight} \
                --adaptation cdan \
                --aug_rotation_prob 1.0 \
                --num_avgpool_head 1  \
                --seed ${SEED} \
                --save_dir ${BASE_DIR} \
                --ignore_dataset bahamas_1.pkl \
                --apply_intrinsic_ell 0. \
                --zl 0.305 \
                --zs ${ZS}

        else
            echo "Done base model"
        fi

        SRC="bahamas_obs"
        TGT="darkskies_obs"

        PREVIOUS_NAME=${BASE_DIR}/${RUN_NAME}_final.pth

        COMMON_ARGS="--batch_size 32 --epochs 10 --lr 0.0001 --eval_interval 2 --project_name incremental_learning --image_size 100 --use_wandb"
        COMMON_ARGS=$COMMON_ARGS" --model squeezenet1_1 --pretrained --mass_index 0 --in_channels 2 --verbose --weighting_scheme inverse_frequency"


    
        TUNE_DIR=models/${FILTER}/
        
        OUTPUT=${RUN_NAME}_ft_zs1p65

        for intrinsic_ell in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
        do

            

            python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT \
                --model squeezenet1_1 --pretrained \
                --weighting_scheme inverse_frequency \
                --run_name ${OUTPUT} --checkpoint ${PREVIOUS_NAME} \
                --adaptation_weight ${adaptation_weight} --adaptation cdan \
                --aug_rotation_prob 1.0 \
                --apply_intrinsic_ell $intrinsic_ell \
                --num_avgpool_head 1  \
                --seed $SEED \
                --zl 0.305 \
                --zs ${ZS} \
                --jwst_filter ${FILTER} \
                --save_dir ${TUNE_DIR} \
                --ignore_dataset bahamas_1.pkl


            PREVIOUS_NAME=${TUNE_DIR}/${OUTPUT}_best.pth 

            

        done

        COMMON_ARGS="--batch_size 32 --epochs 20 --lr 0.0001 --eval_interval 2 --project_name incremental_learning --image_size 100 --use_wandb"
        COMMON_ARGS=$COMMON_ARGS" --model squeezenet1_1 --pretrained --mass_index 0 --in_channels 2 --verbose --weighting_scheme inverse_frequency"
        
        # the final iteration is to save the most aligned version
        python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT \
                --model squeezenet1_1 --pretrained \
                --weighting_scheme inverse_frequency \
                --run_name ${OUTPUT} --checkpoint ${PREVIOUS_NAME} \
                --adaptation_weight ${adaptation_weight} --adaptation cdan \
                --aug_rotation_prob 1.0 \
                --apply_intrinsic_ell 1.0 \
                --num_avgpool_head 1  \
                --seed $SEED \
                --zl 0.305 \
                --zs ${ZS} \
                --jwst_filter ${FILTER} \
                --save_dir ${TUNE_DIR} \
                --ignore_dataset bahamas_1.pkl \
                --save_most_aligned


        PREVIOUS_NAME=${TUNE_DIR}/${OUTPUT}_best.pth 
            
            
            

    done
done