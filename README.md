# PurposeToM
This is an official implementation of PurposeToM.

## Prepare Packages
To install packages, run the following command.
```
cd PurposeToM
pip install -r requirements.txt
```

## Evaluate Methods including PurposeToM
To evaluate different methods, run the following command.

You may change "method" argument to one of {base, cot, ps, simulation, ptom, ours} to evaluate different methods.
```
cd code

# GPT family / BigToM
bash run_bigtom_gpt.sh

# Llama family / BigToM
bash run_bigtom_gpt.sh

# GPT family / BigToM
bash run_tomi_gpt.sh

# Llama family / ToMi
bash run_tomi.sh

```
