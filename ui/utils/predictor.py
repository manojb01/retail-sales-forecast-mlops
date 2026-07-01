"""
Improved predictor combining FeatureEngineer consistency with iterative forecasting.
This fixes the sales=0 bug by predicting day-by-day and using predictions as lag features.
"""

import pandas as pd
import numpy as np
import sys
import logging
from datetime import timedelta
from typing import Dict, Any, Tuple, Optional

# Add include path
sys.path.append("/usr/local/airflow/include")

from feature_engineering.feature_pipeline import FeatureEngineer

logger = logging.getLogger(__name__)


class ImprovedPredictor:
    """
    Improved predictor that combines:
    1. FeatureEngineer for consistency with training
    2. Iterative forecasting to avoid sales=0 contamination
    3. Confidence intervals and error handling from SimplePredictor
    """

    def __init__(self, model, scalers, encoders, feature_cols):
        """
        Initialize predictor with trained model artifacts.

        Args:
            model: Trained model (XGBoost, LightGBM, etc.)
            scalers: Dict of scalers from training
            encoders: Dict of encoders from training
            feature_cols: List of feature column names from training
        """
        self.model = model
        self.scalers = scalers
        self.encoders = encoders
        self.feature_cols = feature_cols
        self.feature_engineer = FeatureEngineer()

    def predict_iterative(
        self,
        historical_data: pd.DataFrame,
        forecast_days: int,
        include_confidence: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Make predictions using iterative forecasting approach.

        This method predicts day-by-day, using each prediction as input
        for subsequent predictions. This ensures lag features reference
        actual predictions rather than zeros.

        Args:
            historical_data: DataFrame with historical sales data
            forecast_days: Number of days to forecast
            include_confidence: Whether to include confidence intervals

        Returns:
            Tuple of (predictions_df, error_message)
            predictions_df has columns: date, store_id, predicted_sales,
                                        lower_bound, upper_bound
        """
        try:
            logger.info(f"Starting iterative forecast for {forecast_days} days")

            # Validate inputs
            if historical_data is None or len(historical_data) == 0:
                return None, "Historical data is empty"

            if forecast_days <= 0:
                return None, "Forecast days must be positive"

            # Prepare historical data
            hist_data = historical_data.copy()
            hist_data['date'] = pd.to_datetime(hist_data['date'])
            hist_data = hist_data.sort_values(['store_id', 'date'])

            # Get stores and last date
            stores = hist_data['store_id'].unique()
            last_date = hist_data['date'].max()

            logger.info(f"Historical data: {len(hist_data)} records, "
                       f"{len(stores)} stores, last date: {last_date}")

            # Apply feature engineering to historical data
            hist_features = self.feature_engineer.create_all_features(
                hist_data,
                target_col='sales',
                date_col='date',
                group_cols=['store_id'],
                categorical_cols=['store_id']
            )

            # Start with historical data in working dataset
            working_data = hist_data.copy()
            predictions_list = []

            # Predict day by day
            for day_offset in range(1, forecast_days + 1):
                logger.debug(f"Predicting day {day_offset}/{forecast_days}")

                # Create future date
                future_date = last_date + timedelta(days=day_offset)

                # Create future records for each store
                future_records = []
                for store in stores:
                    store_hist = working_data[working_data['store_id'] == store]

                    # Use day-of-week patterns for more realistic auxiliary features
                    future_dow = future_date.dayofweek

                    # Get same day-of-week data from last 4 weeks
                    same_dow_data = store_hist[
                        store_hist['date'].dt.dayofweek == future_dow
                    ].tail(4)

                    if len(same_dow_data) > 0:
                        # Use day-of-week average for more realistic patterns
                        avg_traffic = same_dow_data['customer_traffic'].mean()
                        avg_qty = same_dow_data['quantity_sold'].mean()
                        avg_profit = same_dow_data['profit'].mean()
                    else:
                        # Fallback to overall average
                        avg_traffic = store_hist['customer_traffic'].tail(7).mean()
                        avg_qty = store_hist['quantity_sold'].tail(7).mean()
                        avg_profit = store_hist['profit'].tail(7).mean()

                    future_records.append({
                        'date': future_date,
                        'store_id': store,
                        'sales': 0,  # Temporary placeholder - will be replaced with prediction
                        'quantity_sold': avg_qty,
                        'profit': avg_profit,
                        'has_promotion': 0,  # Can be parameterized if needed
                        'customer_traffic': avg_traffic,
                        'is_holiday': 0  # Can be enhanced with holiday detection
                    })

                future_df = pd.DataFrame(future_records)

                # Combine working data with this future day
                combined_df = pd.concat([working_data, future_df], ignore_index=True)
                combined_df = combined_df.sort_values(['store_id', 'date'])

                # Apply feature engineering to combined data
                combined_features = self.feature_engineer.create_all_features(
                    combined_df,
                    target_col='sales',
                    date_col='date',
                    group_cols=['store_id'],
                    categorical_cols=['store_id']
                )

                # Extract only the future rows (for this day)
                future_features = combined_features[
                    combined_features['date'] == future_date
                ].copy()

                # Select and prepare features for prediction
                X_future = self._prepare_features(future_features)

                # Make predictions
                predictions = self.model.predict(X_future)

                # Ensure predictions are positive
                predictions = np.maximum(predictions, 0)

                # Store predictions with metadata
                for idx, (_, row) in enumerate(future_features.iterrows()):
                    pred_value = predictions[idx]
                    predictions_list.append({
                        'date': future_date,
                        'store_id': row['store_id'],
                        'predicted_sales': pred_value
                    })

                    # Add this prediction to working data for next iteration
                    # This is the key: future lag features will reference this prediction
                    working_data = pd.concat([
                        working_data,
                        pd.DataFrame([{
                            'date': future_date,
                            'store_id': row['store_id'],
                            'sales': pred_value,  # Use PREDICTION, not zero!
                            'quantity_sold': row.get('quantity_sold', avg_qty),
                            'profit': row.get('profit', avg_profit),
                            'has_promotion': row.get('has_promotion', 0),
                            'customer_traffic': row.get('customer_traffic', avg_traffic),
                            'is_holiday': row.get('is_holiday', 0)
                        }])
                    ], ignore_index=True)

            # Convert predictions to DataFrame
            predictions_df = pd.DataFrame(predictions_list)

            # Add confidence intervals
            if include_confidence:
                predictions_df = self._add_confidence_intervals(
                    predictions_df,
                    historical_data
                )

            # Calculate summary statistics
            summary = self._calculate_summary(predictions_df)
            logger.info(f"Forecast complete: {summary}")

            return predictions_df, None

        except Exception as e:
            error_msg = f"Error in iterative prediction: {str(e)}"
            logger.error(error_msg, exc_info=True)
            import traceback
            return None, f"{error_msg}\n{traceback.format_exc()}"

    def _prepare_features(self, feature_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction: encoding, selection, scaling.

        Args:
            feature_df: DataFrame with engineered features

        Returns:
            Numpy array ready for model prediction
        """
        # Select only features that model was trained on
        available_features = [
            col for col in self.feature_cols
            if col in feature_df.columns
        ]

        if len(available_features) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available_features)
            logger.warning(f"Missing features: {missing}")

        X = feature_df[available_features].copy()

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.encoders:
                try:
                    X[col] = self.encoders[col].transform(X[[col]]).ravel()
                except Exception as e:
                    logger.warning(f"Error encoding {col}: {e}, using default")
                    X[col] = 0

        # Ensure exact column order
        X = X[self.feature_cols]

        # Apply scaling
        if 'standard' in self.scalers:
            try:
                X_scaled = self.scalers['standard'].transform(X)
                X = pd.DataFrame(
                    X_scaled,
                    columns=self.feature_cols,
                    index=X.index
                )
            except Exception as e:
                logger.warning(f"Error scaling features: {e}")

        return X.values

    def _add_confidence_intervals(
        self,
        predictions_df: pd.DataFrame,
        historical_data: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Add confidence intervals based on historical prediction variance.

        For now, uses a simple approach based on historical volatility.
        Can be enhanced with prediction intervals from the model.

        Args:
            predictions_df: DataFrame with predictions
            historical_data: Historical data for variance estimation
            confidence_level: Confidence level (default 95%)

        Returns:
            DataFrame with lower_bound and upper_bound columns added
        """
        # Calculate historical coefficient of variation per store, capped to keep CI sensible
        cv_by_store = {}
        for store in historical_data['store_id'].unique():
            store_data = historical_data[
                historical_data['store_id'] == store
            ]['sales']
            raw_cv = store_data.std() / store_data.mean() if store_data.mean() > 0 else 0.15
            cv_by_store[store] = min(raw_cv, 0.30)  # cap at 30% to prevent absurdly wide bands

        # Add confidence intervals
        predictions_df['lower_bound'] = predictions_df.apply(
            lambda row: max(0, row['predicted_sales'] * (
                1 - cv_by_store.get(row['store_id'], 0.15) * 1.96
            )),
            axis=1
        )
        predictions_df['upper_bound'] = predictions_df.apply(
            lambda row: row['predicted_sales'] * (
                1 + cv_by_store.get(row['store_id'], 0.15) * 1.96
            ),
            axis=1
        )

        return predictions_df

    def _calculate_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for predictions.

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_records': len(predictions_df),
            'total_predicted_sales': predictions_df['predicted_sales'].sum(),
            'avg_daily_sales': predictions_df['predicted_sales'].mean(),
            'min_daily_sales': predictions_df['predicted_sales'].min(),
            'max_daily_sales': predictions_df['predicted_sales'].max(),
            'stores': predictions_df['store_id'].nunique()
        }

    def predict_batch(
        self,
        historical_data: pd.DataFrame,
        forecast_days: int
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Alias for predict_iterative for backward compatibility.
        """
        return self.predict_iterative(historical_data, forecast_days)
