
### Requirements
  * `pip install recbole`
  * `pip install causal-conv1d>=1.4.0`
  * `pip install mamba-ssm`
  

### Run MaSwaRec
Please run this code in RecBole

```python run_recbole.py --model=MaSwaRec --dataset=ml-1m  --config_files=ml.yaml```  

```python run_recbole.py --model=MaSwaRec --dataset=amazon-beauty --config_files=amazon.yaml```

```python run_recbole.py --model=MaSwaRec --dataset=amazon-office-products --config_files=amazon.yaml```

```python run_recbole.py --model=MaSwaRec --dataset=amazon-baby --config_files=amazon.yaml```

```python run_recbole.py --model=MaSwaRec --dataset=amazon-musical-instruments  --config_files=amazon.yaml```  




## Acknowledgment

This project is based on [Mamba](https://github.com/state-spaces/mamba), [Causal-Conv1d](https://github.com/Dao-AILab/causal-conv1d), and [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their excellent works.
