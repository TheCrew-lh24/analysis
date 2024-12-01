root=$1
input=$2
mode=$3

mkdir $root

mkdir $root/0_init
python stage_1.py -i $input -s init -o $root/0_init/init.csv

mkdir $root/1_merge_iban
python stage_1.py -i $root/0_init/init.csv -s match_iban -a split -n 1 -o $root/1_merge_iban
python stage_1.py -i $root/1_merge_iban -s match_iban -a execute -o $root/1_merge_iban -n 0
python stage_1.py -i $root/0_init/init.csv -s match_iban -a merge -n 1 -o $root/1_merge_iban

if [ $mode == "train" ]; then
    python stage_1.py -i $root/1_merge_iban/merged.csv -s eval
fi
if [ $mode == "test" ]; then
    python stage_1.py -i $root/1_merge_iban/merged.csv -s make_submission -o $root/1_merge_iban/submission.csv
fi

mkdir $root/2_merge_phone
python stage_1.py -i $root/1_merge_iban/merged.csv -s match_phone -a split -n 1 -o $root/2_merge_phone
python stage_1.py -i $root/2_merge_phone -s match_phone -a execute -o $root/2_merge_phone -n 0
python stage_1.py -i $root/1_merge_iban/merged.csv -s match_phone -a merge -n 1 -o $root/2_merge_phone

if [ $mode == "train" ]; then
    python stage_1.py -i $root/2_merge_phone/merged.csv -s eval
fi
if [ $mode == "test" ]; then
    python stage_1.py -i $root/2_merge_phone/merged.csv -s make_submission -o $root/2_merge_phone/submission.csv
fi

mkdir $root/3_merge_name
python stage_1.py -i $root/2_merge_phone/merged.csv -s match_name -a split -n 1 -o $root/3_merge_name
python stage_1.py -i $root/3_merge_name -s match_name -a execute -o $root/3_merge_name -n 0
python stage_1.py -i $root/2_merge_phone/merged.csv -s match_name -a merge -n 1 -o $root/3_merge_name

if [ $mode == "train" ]; then
    python stage_1.py -i $root/3_merge_name/merged.csv -s eval
fi
if [ $mode == "test" ]; then
    python stage_1.py -i $root/3_merge_name/merged.csv -s make_submission -o $root/3_merge_name/submission.csv
fi
