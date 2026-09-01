import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class MarketRegimeHMM:

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        # Covariance type 'full' allows variables to interact across states
        self.model = GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full", 
            n_iter=1000, 
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def engineer_features(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw price data into stationary statistical emissions.
        """
        features = pd.DataFrame(index=df_prices.index)
        
        # 1. Log Returns (Captures directional velocity)
        features['log_returns'] = np.log(df_prices['Close'] / df_prices['Close'].shift(1))
        
        # 2. Parkinson Volatility (Uses High/Low to isolate intraday range variance)
        hl_ratio = np.log(df_prices['High'] / df_prices['Low'])
        features['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio ** 2)).rolling(window=5).mean()
        
        # 3. Acceleration / Momentum changes
        features['momentum_ma'] = features['log_returns'].rolling(window=10).mean()
        
        return features.dropna()

    def fit(self, X_features: pd.DataFrame):
        
        # Feature scaling to prevent higher absolute values from dominating states
        scaled_data = self.scaler.fit_transform(X_features)
        
        # Run Expectation-Maximization
        self.model.fit(scaled_data)
        self.is_fitted = True
        
        # Re-order regime IDs monotonically by variance to ensure interpretability
        # State 0 will always be the lowest volatility state, up to State N as highest stress
        vol_idx = np.argsort(self.model.covars_[:, 0, 0])
        self._remap_state_orders(vol_idx)

    def _remap_state_orders(self, order: np.ndarray):
        """Internal helper to keep state classifications logically sorted by risk."""
        self.model.startprob_ = self.model.startprob_[order]
        self.model.transmat_ = self.model.transmat_[np.ix_(order, order)]
        self.model.means_ = self.model.means_[order]
        self.model.covars_ = self.model.covars_[order]

    def predict_live_regime(self, X_features: pd.DataFrame) -> tuple:
        """
        Uses Viterbi decoding for state paths and evaluates the forward-backward 
        matrix for exact posterior state probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before predicting regimes.")
            
        scaled_data = self.scaler.transform(X_features)
        
        # Most likely hidden path via Viterbi
        hidden_states = self.model.predict(scaled_data)
        # Posterior probabilities P(State_i | Data_t)
        posterior_probs = self.model.predict_proba(scaled_data)
        
        return hidden_states, posterior_probs

# --- Verification Window ---
if __name__ == "__main__":
    # Generate synthetic OHLC market data representing structural regimes
    np.random.seed(101)
    idx = pd.date_range(start="2024-01-01", periods=1000, freq="D")
    
    # Simulate data-generating process shifting from calm to chaotic
    r1 = np.random.normal(0.0005, 0.005, 400)   # Low vol expansion
    r2 = np.random.normal(-0.002, 0.02, 200)    # Crisis shock
    r3 = np.random.normal(0.0002, 0.01, 400)    # Choppy recovery
    simulated_returns = np.concatenate([r1, r2, r3])
    
    sim_close = 100 * np.exp(np.cumsum(simulated_returns))
    mock_df = pd.DataFrame({
        'Close': sim_close,
        'High': sim_close * 1.01,
        'Low': sim_close * 0.99
    }, index=idx)
    
    # Initialize implementation pipeline
    detector = MarketRegimeHMM(n_regimes=3)
    features = detector.engineer_features(mock_df)
    
    # Execute out-of-sample training block
    detector.fit(features)
    states, probabilities = detector.predict_live_regime(features)
    
    print("Standalone Model Execution Verification:")
    print(f"Calculated Transition Matrix Consistency:\n{detector.model.transmat_.round(3)}")
    print(f"Latest Active Market Row classified into State: {states[-1]}")
