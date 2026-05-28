# Used-Car Auction ML Pipeline and Bidding Agent

This project contains a complete training pipeline and deterministic bidding agent for wholesale used-car auctions.

## Files

- `analysis_Daksh.ipynb`: EDA, cleaning rationale, feature engineering rationale, and model workflow.
- `preprocessing_Daksh.py`: Fitted preprocessing/encoding logic shared by training and inference.
- `train_model_Daksh.py`: Trains, tunes, evaluates, and saves the model artifacts.
- `model_Daksh.pkl`: Generated after training.
- `encoders_Daksh.pkl`: Generated after training. Required by the live agent.
- `metrics_Daksh.csv`: Generated validation comparison for tuned versus default model.
- `agent_Daksh.py`: Live auction bidding agent.

## Run

Place `car_auction_train.csv` in this folder, then run:

```powershell
python train_model_Daksh.py
```

The training script saves `model_Daksh.pkl`, `encoders_Daksh.pkl`, and `metrics_Daksh.csv` using relative paths.

## Agent Contract

The simulator should instantiate:

```python
from agent_Daksh import Agent

agent = Agent(bankroll=500_000)
value = agent.evaluate_car(car_dict)
bid = agent.place_bid(current_highest_bid=10_000, auction_round=3)
```

`evaluate_car` accepts a single car dictionary. It does not compute row-dependent means, group statistics, or any training-data aggregations during live evaluation. All imputations, outlier caps, and category levels are loaded from `encoders_Daksh.pkl`.
