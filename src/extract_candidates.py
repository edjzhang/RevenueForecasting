import numpy as np
from fastdtw import fastdtw
from math import sqrt

class CandidateExtractor:
    """ This class can be used to identify the time series subsequence for each potential comparison company 
    with the smallest distance from the series to forecast that fits specified constraints
    
    Initial inputs:
        ts_to_match (np.array): contains the time series to forecast, monthly data points
        ts_candidates (list of np.arrays): contains the time series for the potential company comparisons, monthly data points
        candidate_names (list of str): contains the names of the candidates corresponding to ts_candidates, default numeric index
    """
    
    def __init__(self, ts_to_match, ts_candidates, candidate_names = None):
        self.forecast_lag_months = 12
        
        self.ts_to_match = ts_to_match
        # Assign names to track original candidate order relative to any sorted lists
        if not candidate_names:
            candidate_names = np.array(range(len(ts_candidates)))
            
        # Remove any candidates that are not long enough (and their corresponding names)
        ts_to_keep = np.where([len(ts) >= len(ts_to_match) + self.forecast_lag_months for ts in ts_candidates])[0]
        ts_candidates = np.array(ts_candidates)[ts_to_keep]
        self.candidate_names = np.array(candidate_names)[ts_to_keep]
        
        # Remove the last year of data points from each candidate since we do not have forecasts for those
        self.ts_candidates = np.array([ts[:-self.forecast_lag_months] for ts in ts_candidates])
        # Get values to forecast from same time series, excluding the length of the time series to predict and forecast lag
        self.forecast_vals = np.array([ts[len(ts_to_match) + self.forecast_lag_months - 1:] for ts in ts_candidates])
        
        # Set default indices for when `filter_ts_candidates` function is not used
        self.start_indices = [0] * len(ts_candidates)
        self.end_indices = [len(ts) for ts in ts_candidates]
        
    def filter_ts_candidates(self, min_val, max_val):
        """ Remove subsequences that do not meet criteria
        
        Inputs:
            min_val (int): time series subsequence must start and remain above this value
            max_val (int): time series subsequence must not go above this value
        """
        self.start_indices = []
        self.end_indices = []
        for ts in self.ts_candidates:
            try:
                start_idx = np.where(ts < min_val)[0][-1] + 1
            except IndexError:
                start_idx = 0
            try:
                end_idx = np.where(ts > max_val)[0][0]
            except IndexError:
                end_idx = len(ts)
            self.start_indices.append(start_idx)
            self.end_indices.append(end_idx)
        
        self.forecast_vals = [val[start_idx:end_idx] for val, start_idx, end_idx in zip(self.forecast_vals, 
                                                                                        self.start_indices,
                                                                                        self.end_indices)]
        self.ts_candidates = [ts[start_idx:end_idx] for ts, start_idx, end_idx in zip(self.ts_candidates, 
                                                                                      self.start_indices,
                                                                                      self.end_indices)]
        
        # Remove any candidates that are not long enough (and their corresponding names/forecasts) after filtering
        ts_to_keep = np.where([len(ts) >= len(self.ts_to_match) for ts in self.ts_candidates])[0]
        self.ts_candidates = np.array(self.ts_candidates)[ts_to_keep]
        self.candidate_names = np.array(self.candidate_names)[ts_to_keep]
        self.forecast_vals = np.array(self.forecast_vals)[ts_to_keep]
    
    def get_best_candidates(self, num_comps, norm = False, dtw = False):
        """ Extract the time series with the lowest distance to the time series to forecast
        
        Inputs:
            num_comps (int): number of time series subsequences to return, one per company comparison
            norm (bool): if True, normalize time series to forecast and candidates when calculating distance
            dtw (bool): if True, use dynamic-time-warping distance metric; else, use Euclidean default
        
        Returns:
            best_candidates (list of np. arrays): len of num_comps, contains best subsequences with same length as ts_to_match,
                                                  one per potential company comparison, sorted ascending by distance
            best_distances (list of floats): contains the distance values for each candidate
            best_indices (list of int duples): contains the indices of the best_candidates, to sync with other features (start, end)
            forecast_vals (list of floats): contains values to be used as truth for training forecasting model
            candidate_order (list of str): sorted to correspond to the order of the time series in best_candidates
        """
        # Generate all possible stepwise time series with the length of `ts_to_match`
        expanded_ts_candidates = []
        for ts, candidate_name in zip(self.ts_candidates, self.candidate_names):
            expanded_ts_candidates.append([ts[x:len(self.ts_to_match)+x] for x in range(len(ts)-len(self.ts_to_match)+1)])
        
        # Calculate distance, normalizing if specified, and get ts, indices, forecast values corresponding to the best distance
        best_candidates, best_distances, best_indices, forecast_vals = [], [], [], []
        for expansion, forecast_val, start_idx in zip(expanded_ts_candidates, self.forecast_vals, self.start_indices):
            if dtw:
                if norm:
                    dist = [sqrt(fastdtw(ts / ts[0], self.ts_to_match / self.ts_to_match[0])[0]) for ts in expansion]
                else:
                    dist = [sqrt(fastdtw(ts, self.ts_to_match)[0]) for ts in expansion]
            else:
                if norm:
                    dist = [np.linalg.norm(ts / ts[0] - self.ts_to_match / self.ts_to_match[0]) for ts in expansion] 
                else:
                    dist = [np.linalg.norm(ts - self.ts_to_match) for ts in expansion]
            best_candidate_idx = np.argmin(dist)
            best_candidates.append(expansion[best_candidate_idx])
            best_distances.append(np.min(dist))
            best_indices.append((start_idx + best_candidate_idx, 
                                 start_idx + best_candidate_idx + len(self.ts_to_match)))
            forecast_vals.append(forecast_val[best_candidate_idx])
        
        # Sort by the best distances, and return top num_comps time series and corresponding info
        sorted_tups = zip(*sorted(zip(best_distances, best_candidates, best_indices, forecast_vals, self.candidate_names)))
        best_distances, best_candidates, best_indices, forecast_vals, candidate_order = (list(tup)[:num_comps] for tup in sorted_tups)

        return best_candidates, best_distances, best_indices, forecast_vals, candidate_order