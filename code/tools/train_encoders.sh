#!/usr/bin/env bash
#conda activate lmdetect

shopt -s nullglob
src=$1
dst=$2
gpu_id=$3
every=$4
for file in $src/zs_and_nr_*.json; do
    fn=$(basename $file);
    config_id=$(grep -o '[0-9]*' <<< $fn)
    if (($config_id % 16 == $every))
    then
        # 68 LM
        encoder_fn="encoder_68_$config_id.torch"
        encoder_path="$dst/$encoder_fn"
        if [ -f $encoder_path ]; then
            echo "$encoder_path already exists"
        else
            cmd="python ../pdm/encoder.py $file $encoder_path --gpu $gpu_id"
            #printf "\n---------------\n"
            echo $cmd
            $cmd
            #printf "\n---------------\n\n"
        fi

        # 49 LM
        encoder_fn="encoder_49_$config_id.torch"
        encoder_path="$dst/$encoder_fn"
        if [ -f $encoder_path ]; then
            echo "$encoder_path already exists"
        else
            cmd="python ../pdm/encoder.py $file $encoder_path --gpu $gpu_id --is_49lm"
            #printf "\n---------------\n"
            echo $cmd
            $cmd
            #printf "\n---------------\n\n"
        fi
    #else
    #    printf "\nSkip $config_id"
    fi
done
shopt -u nullglob