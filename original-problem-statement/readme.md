# 🛺 The "Namma Auto" Logistics Challenge
### *The shortest distance is a straight line. But not when you are dodging cows and tourists.*

---

## 1. Executive Summary
In Mysore, the Auto Rickshaw is the lifeline of the city. However, pricing and time estimation remain a chaotic mix of negotiation and guesswork.

You are acting as a **Lead Data Scientist for the Mysore City Transport Division**. You have been provided with raw telemetry data from a pilot project involving thousands of digitised Auto Rickshaws.

Your objective is to build a "Fair Time Engine." The meter is ticking, but the traffic is unpredictable. If your model fails, the meter runs too high (angry customers) or too low (angry drivers).

---

## 2. The Objective
Your task is to model the total duration of auto trips. The raw data provides the *What* (Coordinates) and the *When* (Timestamps). Your model must figure out the *How*.

To succeed, you must:

1.  **Translate Space:** You are given raw coordinates. But an Auto driver knows that 1 km in **Devaraja Market** is very different from 1 km on the **Ring Road**. Your model needs to understand the "texture" of the city geography.
2.  **Decode Time:** Mysore has a specific rhythm—Dasara traffic, weekend tourist spikes, and school rush hours. You must extract these invisible signals from the timestamps.
3.  **Sanity Check Reality:** The dataset reflects real-world chaos. Meters break. GPS drifts. Drivers take naps. You must identify and handle data that defies the laws of physics.
4.  **Forecast:** Build a Regression model to predict the `trip_duration` (in seconds) for the test set.

---

## 3. Dataset Description

The dataset covers individual Auto Rickshaw trips. The variables are raw and require significant engineering to be useful.

### File: `train.csv` (Historical Auto Data)

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

## 4. Key Challenges to Solve

We are not telling you *how* to process the data. You must investigate the following conceptual problems:

1.  **The Geometry of Travel:** Coordinates give you a straight line on a map. But Autos move through a grid of streets, turns, and one-ways. Does the "Crow Flies" distance actually correlate with travel time, or is there a better mathematical representation of urban movement?
2.  **The "Pulse" of the City:** Time is not linear in traffic. A trip at 2:00 AM is structurally different from a trip at 6:00 PM. How will you represent this cyclicity to a mathematical model?
3.  **The "Black Holes" of Data:** Some trips in this dataset appear to traverse the entire city in seconds, while others move nowhere for hours. If you feed this noise to your model, it will fail. How will you detect "Impossible Trips"?
4.  **Regional Dynamics:** Is a Pickup at the Railway Station the same as a Pickup in a quiet residential layout? How can you group continuous coordinates into meaningful "Zones" of activity?

---

## 5. Judging Criteria

We are looking for **Pattern Recognition** and **Engineering Intuition**.

| Metric | Description |
| :--- | :--- |
| **Feature Engineering** | Did you successfully translate raw coordinates into meaningful distance and direction vectors? |
| **Spatial Analytics** | Did you identify high-density zones? Did your model treat the "Heritage Core" differently from the "Outskirts"? |
| **Data Integrity** | How did you handle the "Impossible Trips"? Did you use a statistical approach or a domain-logic approach? |
| **Modeling Strategy** | Did you choose a metric that accurately penalizes errors relative to the trip length? (e.g. Being wrong by 5 mins on a 10 min trip is worse than on a 2-hour trip). |

---

## 6. Submission Requirements

You must submit a **Jupyter Notebook** (`.ipynb`) containing:

1.  **EDA & Visualization:** Maps or charts showing the spatial distribution of pickups/dropoffs.
2.  **Feature Engineering Log:** A section explaining the derived features you created.
3.  **Modeling & Evaluation:** Your training process and final validation score.

---

### 💡 Pro-Tips for Participants
* **Leakage Warning:** You have the Dropoff Timestamp in the training set. If you use this variable to predict duration, you are effectively predicting the answer using the answer. This is forbidden.
* **Coordinate Math:** Latitude and Longitude are not linear numbers like "Price" or "Age". You cannot simply subtract them to find distance.
* **Visualize:** Plot the pickups. Do they look like a map of Mysore? If they look like a random scatter, you might need to clean the outliers first.