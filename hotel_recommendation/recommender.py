# hotel_recommendation/recommender.py

import pandas as pd

class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, place, days, budget, topn=5):
        filtered_hotels = self.items_df[
            (self.items_df["place"] == place) &
            (self.items_df["days"] == days) &
            (self.items_df["price"] <= budget)
        ]
        if filtered_hotels.empty:
            return pd.DataFrame()

        recommendations_df = filtered_hotels.groupby("name")["price"].min().reset_index()
        recommendations_df = recommendations_df.sort_values(by="price", ascending=True).head(topn)
        return recommendations_df
