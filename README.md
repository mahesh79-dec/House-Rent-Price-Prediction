# House Rent Price Prediction Using Machine Learning

## Included
- `app.py`: Streamlit web interface for rent prediction
- `train_model.py`: training and evaluation script
- `data/house_rent_data.csv`: 650-row synthetic demonstration dataset
- `models/`: location for the trained model file

## Important data note
The included CSV is **synthetic demo data generated for this academic project**. It is not actual rental-market data. Model scores and predictions are not evidence of real-world accuracy. For a real-market project, replace the CSV with a documented, legally obtained dataset and retrain the model.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Train the model:
   ```bash
   python train_model.py
   ```
6. Launch the web app:
   ```bash
   streamlit run app.py
   ```

The app may also train the model automatically the first time it is launched.

## Model
A scikit-learn Pipeline combines OneHotEncoder and RandomForestRegressor. The data is split into 80% training and 20% testing. Evaluation reports MAE, RMSE and R².

## Suggested project report structure
1. Title page
2. Certificate / declaration (only if required by your department)
3. Abstract
4. Introduction
5. Problem statement
6. Objectives and scope
7. Literature review
8. System requirements
9. Methodology and system architecture
10. Dataset description
11. Implementation
12. Results and discussion
13. Limitations
14. Future scope
15. Conclusion
16. References
