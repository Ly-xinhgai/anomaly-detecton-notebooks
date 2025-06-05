import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats
from scipy.stats import ttest_ind
import math

# 1. SPIKE DETECTION ALGORITHMS

def z_score(values: List[float], threshold: float = 2.0) -> List[bool]:
    if len(values) < 2:
        return [False] * len(values)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # dùng sample std (n-1)
    
    if std == 0:
        return [False] * len(values)
    
    z_scores = [(x - mean) / std for x in values]
    return [abs(z) > threshold for z in z_scores]

def modified_z_score(values: List[float], threshold: float = 3.5) -> List[bool]:
    if len(values) < 2:
        return [False] * len(values)
    
    median = np.median(values)
    mad = np.median([abs(x - median) for x in values])
    
    if mad == 0:
        return [False] * len(values)
    
    modified_z_scores = [0.6745 * (x - median) / mad for x in values]
    return [abs(z) > threshold for z in modified_z_scores]

def moving_z_score(values: List[float], window: int = 30, threshold: float = 3.0) -> List[bool]:
    if len(values) < window:
        return z_score(values, threshold)
    
    anomalies = [False] * len(values)
    
    for i in range(window - 1, len(values)):
        window_data = values[max(0, i - window + 1):i + 1]
        mean = sum(window_data) / len(window_data)
        variance = sum((x - mean) ** 2 for x in window_data) / (len(window_data) - 1)
        std_dev = variance ** 0.5
        
        if std_dev > 0:
            z = abs((values[i] - mean) / std_dev)
            anomalies[i] = z > threshold
    
    return anomalies

# 2. CHANGE POINT DETECTION ALGORITHMS

def cusum(data: List[float], threshold: float = 5.0, drift: float = 0.01) -> List[Tuple[int, float]]:
    if len(data) < 2:
        return []
    
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / (len(data) - 1)) ** 0.5
    
    # Normalize threshold and drift by standard deviation
    h = threshold * std_dev if std_dev > 0 else threshold
    k = drift * std_dev if std_dev > 0 else drift
    
    pos_cusum, neg_cusum = 0, 0
    anomalies = []

    for i, x in enumerate(data):
        pos_cusum = max(0, pos_cusum + (x - mean - k))
        neg_cusum = max(0, neg_cusum + (mean - x - k))
        
        if pos_cusum > h or neg_cusum > h:
            anomalies.append((i, x))
            pos_cusum, neg_cusum = 0, 0  # reset

    return anomalies

def page_hinkley(data: List[float], delta: float = 1.0, threshold: float = 10.0) -> List[int]:
    if len(data) < 2:
        return []
    
    mean = sum(data) / len(data)
    m = 0
    M = 0
    change_points = []
    
    for i, x in enumerate(data):
        m = m + (x - mean - delta/2)
        M = max(0, M + m)
        
        if M > threshold:
            change_points.append(i)
            M = 0
            m = 0
    
    return change_points

def ewma_control_chart(data: List[float], lambda_param: float = 0.2, 
                      threshold_factor: float = 3.0) -> List[bool]:
    if len(data) < 2:
        return [False] * len(data)
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    std_dev = variance ** 0.5
    
    ewma = [data[0]]
    anomalies = [False]
    
    for i in range(1, len(data)):
        ewma_val = lambda_param * data[i] + (1 - lambda_param) * ewma[i-1]
        ewma.append(ewma_val)
        
        # Control limits
        sigma_ewma = std_dev * math.sqrt(lambda_param * (1 - (1-lambda_param)**(2*i)) / (2 - lambda_param))
        ucl = mean + threshold_factor * sigma_ewma
        lcl = mean - threshold_factor * sigma_ewma
        
        anomalies.append(ewma_val > ucl or ewma_val < lcl)
    
    return anomalies

def detect_mean_shift_at_known_points(data: List[float], change_points: List[int], 
                                    window: int = 20, alpha: float = 0.05) -> List[Tuple[int, float]]:
    significant_changes = []
    
    for cp in change_points:
        # FIX: Đảm bảo có đủ dữ liệu cho cả hai bên
        if cp < window or cp + window >= len(data):
            continue
        
        before = data[cp - window:cp]
        after = data[cp:cp + window]
        
        # Kiểm tra xem có đủ dữ liệu và variance > 0
        if len(before) < 2 or len(after) < 2:
            continue
        
        try:
            # Welch's t-test (không giả định equal variance)
            t_stat, p_value = ttest_ind(before, after, equal_var=False)
            
            if p_value < alpha:
                significant_changes.append((cp, p_value))
        except:
            # Nếu t-test fail (ví dụ: variance = 0), skip
            continue
    
    return significant_changes

# 3. TREND DETECTION ALGORITHMS

def detect_increasing_decreasing_sequences(data: List[float], min_len: int = 3) -> List[Tuple[str, int, int]]:
    if len(data) < min_len + 1:
        return []
    
    results = []
    count = 1
    current_direction = None

    for i in range(1, len(data)):
        diff = data[i] - data[i-1]
        
        if diff > 0:
            new_direction = 'increasing'
        elif diff < 0:
            new_direction = 'decreasing'
        else:
            new_direction = 'flat'

        if new_direction == current_direction and new_direction != 'flat':
            count += 1
        else:
            # Kết thúc sequence hiện tại nếu đủ dài
            if current_direction in ('increasing', 'decreasing') and count >= min_len:
                results.append((current_direction, i - count, i - 1))
            
            count = 1 if new_direction != 'flat' else 0
            current_direction = new_direction if new_direction != 'flat' else current_direction

    # Kiểm tra sequence cuối cùng
    if current_direction in ('increasing', 'decreasing') and count >= min_len:
        results.append((current_direction, len(data) - count, len(data) - 1))

    return results

def mann_kendall_trend_test(data: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    n = len(data)
    if n < 3:
        return {'trend': 'no trend', 'p_value': 1.0, 'tau': 0.0}
    
    # Calculate S statistic
    S = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if data[j] > data[i]:
                S += 1
            elif data[j] < data[i]:
                S -= 1
    
    # Calculate variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Calculate Z statistic
    if S > 0:
        z = (S - 1) / math.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / math.sqrt(var_s)
    else:
        z = 0
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Kendall's tau
    tau = S / (n * (n - 1) / 2)
    
    # Determine trend
    if p_value < alpha:
        if tau > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
    else:
        trend = 'no trend'
    
    return {
        'trend': trend,
        'p_value': p_value,
        'tau': tau,
        'S': S,
        'z_score': z
    }

def linear_regression_slope_test(data: List[float], window: int = 30, 
                               threshold: float = 0.1) -> List[Tuple[int, float, bool]]:
    if len(data) < window:
        return []
    
    results = []
    
    for i in range(window - 1, len(data)):
        y = data[i - window + 1:i + 1]
        x = list(range(window))
        
        # Calculate slope
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[j] * y[j] for j in range(n))
        sum_x2 = sum(x[j] ** 2 for j in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Test significance of slope
        is_significant = abs(slope) > threshold
        
        results.append((i, slope, is_significant))
    
    return results


# 4. ENSEMBLE METHODS

def ensemble_voting(data: List[float], methods: Optional[List[str]] = None, 
                   voting_threshold: float = 0.5) -> List[bool]:
    if methods is None:
        methods = ['z_score', 'iqr', 'modified_z_score']
    
    results = []
    
    # Apply each method
    method_results = {}
    if 'z_score' in methods:
        method_results['z_score'] = z_score(data)
    if 'iqr' in methods:
        method_results['iqr'] = iqr_method(data)
    if 'modified_z_score' in methods:
        method_results['modified_z_score'] = modified_z_score(data)
    
    # Voting
    for i in range(len(data)):
        votes = sum(method_results[method][i] for method in method_results)
        total_methods = len(method_results)
        
        results.append(votes / total_methods >= voting_threshold)
    
    return results

def calculate_anomaly_scores(data: List[float], method: str = 'z_score', 
                           **kwargs) -> List[float]:
    if method == 'z_score':
        threshold = kwargs.get('threshold', 3.0)
        mean = sum(data) / len(data)
        std_dev = (sum((x - mean) ** 2 for x in data) / (len(data) - 1)) ** 0.5
        if std_dev == 0:
            return [0.0] * len(data)
        return [abs((x - mean) / std_dev) for x in data]
    
    elif method == 'iqr':
        multiplier = kwargs.get('multiplier', 1.5)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        if iqr == 0:
            return [0.0] * len(data)
        
        scores = []
        for x in data:
            if x < q1:
                scores.append(abs(x - q1) / iqr)
            elif x > q3:
                scores.append(abs(x - q3) / iqr)
            else:
                scores.append(0.0)
        return scores
    
    else:
        raise ValueError(f"Unknown method: {method}")

def evaluate_anomaly_detection(true_anomalies: List[bool], predicted_anomalies: List[bool]) -> Dict[str, float]:
    if len(true_anomalies) != len(predicted_anomalies):
        raise ValueError("Lists must have same length")
    
    tp = sum(t and p for t, p in zip(true_anomalies, predicted_anomalies))
    fp = sum(not t and p for t, p in zip(true_anomalies, predicted_anomalies))
    tn = sum(not t and not p for t, p in zip(true_anomalies, predicted_anomalies))
    fn = sum(t and not p for t, p in zip(true_anomalies, predicted_anomalies))
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / len(true_anomalies)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }
# 6. EXAMPLE USAGE

# if __name__ == "__main__":
#     # Example data with known anomalies
#     np.random.seed(42)
#     normal_data = np.random.normal(100, 10, 100).tolist()
    
#     # Add some spikes
#     normal_data[20] = 150  # spike
#     normal_data[50] = 50   # spike
#     normal_data[80] = 200  # spike
    
#     print("=== SPIKE DETECTION ===")
#     z_anomalies = z_score(normal_data, threshold=2.5)
#     iqr_anomalies = iqr_method(normal_data)
#     mod_z_anomalies = modified_z_score(normal_data)
    
#     print(f"Z-Score detected {sum(z_anomalies)} anomalies")
#     print(f"IQR detected {sum(iqr_anomalies)} anomalies")
#     print(f"Modified Z-Score detected {sum(mod_z_anomalies)} anomalies")
    
#     print("\n=== CHANGE POINT DETECTION ===")
#     # Create data with change point
#     change_data = [100] * 50 + [120] * 50
#     change_points = cusum(change_data, threshold=3.0)
#     print(f"CUSUM detected {len(change_points)} change points: {change_points}")
    
#     print("\n=== TREND DETECTION ===")
#     # Create trending data
#     trend_data = list(range(100)) + [99] * 20  # increasing then flat
#     trends = detect_increasing_decreasing_sequences(trend_data, min_len=10)
#     print(f"Detected trends: {trends}")
    
#     mk_result = mann_kendall_trend_test(list(range(50)))
#     print(f"Mann-Kendall test: {mk_result['trend']} (p={mk_result['p_value']:.4f})