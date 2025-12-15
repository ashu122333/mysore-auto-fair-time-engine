# mysore-auto-fair-time-engine
## Predicting auto-rickshaw trip duration in Mysore using spatial data.
### Problem Statement:
In Mysore, the Auto Rickshaw is the lifeline of the city. However, pricing and time estimation remain a chaotic mix of negotiation and guesswork.
We have been provided with raw telemetry data from a pilot project involving thousands of digitised Auto Rickshaws.
My objective is to build a "Fair Time Engine." The meter is ticking, but the traffic is unpredictable. If your model fails, the meter runs too high (angry customers) or too low (angry drivers).

### Dataset Description:

| Data Field | Description |
| :--- | :--- |
| `id` | A unique identifier for each trip |
| `vendor_id` | Code indicating the Auto Union/Provider ID |
| `pickup_datetime` | Date and time when the meter was engaged |
| `dropoff_datetime` | Date and time when the meter was disengaged |
| `passenger_count` | Number of passengers |
| `pickup_longitude` | The longitude where the meter was engaged |
| `pickup_latitude` | The latitude where the meter was engaged |
| `dropoff_longitude` | The longitude where the meter was disengaged |
| `dropoff_latitude` | The latitude where the meter was disengaged |
| `store_and_fwd_flag` | Boolean flag (Y/N) indicating if data was stored offline |
| `trip_duration` | **(Target)** Duration of the trip in seconds |

---
### Approach:
- spatial sanity checks: Check is the coordinates are real or not 
- physics-based filtering : removing auto with impossible avg speed, distance and time
- distance + zone features
  - Created spatial zone to understand coorilation
  - Calculated Haversin Distance ( required due to earth's curvature )
  - Calculated manhatten distance and roads have grid patterns not a stright line
- time cyclic encoding : Created Time based features ( Month, weekday, hour ) and encoded
- LightGBM modeling : Its a light greadient boosting model which can capture many complex relationship
- RMSLE for fairness : Helps identify comparitive Error

### Key Insights
- distance alone is insufficient, time, day of the week capture more about the roads identity and nature
- evening hours more rush causes more uncertainty
- regional congestion matters

### Results:
- RMSE: ~300 sec ( Mean average error is around 5 min )
- RMSLE : 0.34 ( This is relative error, means the model is fair to short trips )

### Tech Stack:
Python, Pandas, LightGBM, Matplotlib, Scikit-learn

---

### 📁 Repository Structure

```text
mysore-auto-fair-time-engine/
│
├── README.md                      # Project overview & explanation
│
├── mysore-auto-fair-time-engine.ipynb   # Main notebook with full workflow
│
├── src/
│   ├── features.py                # Distance, time, and zone feature functions
│   ├── utils.py                   # Cleaning, sanity checks, helper functions
│   └── train_model.py             # LightGBM/XGBoost training script
│
├── images/                        # Visualizations used in README + notebook
│
├── data/                          # Contains dataset (only if allowed)
│   ├── train.csv 
│   └── test.csv 
│
└── original-problem-statement/
    └── README.md                  # Original problem statement
```

    
