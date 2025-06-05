import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum

# Cấu hình logging
logger = logging.getLogger("kiem-tra")

class AnomalyType(str, Enum):
    # """Loại bất thường có thể phát hiện."""
    SPIKE = "spike"
    CHANGE_MEAN = "change_mean"
    INCREASE = "increase"
    DECREASE = "decrease"
    ALL = "all"

class DetectionAlgorithm(str, Enum):
    # """Thuật toán phát hiện bất thường."""
    ZSCORE = "zscore" 
    IQR = "iqr"
    CUSUM = "cusum"
    MOVING_AVERAGE = "moving_average"
    ENSEMBLE = "ensemble"

class TimeSeriesAnomalyDetector:
    # """Lớp phát hiện bất thường trong dữ liệu chuỗi thời gian."""
    
    def __init__(self, timestamps=None, values=None):
        # Khởi tạo phát hiện bất thường.
        
        # Args:
        #     timestamps: Danh sách thời gian (UNIX timestamp)
        #     values: Danh sách giá trị tương ứng
        self.logger = logging.getLogger(__name__)
        self.set_data(timestamps, values)
    
    def set_data(self, timestamps=None, values=None):
        # Thiết lập dữ liệu chuỗi thời gian.
        
        # Args:
        #     timestamps: Danh sách thời gian (UNIX timestamp)
        #     values: Danh sách giá trị tương ứng
        if timestamps is None or values is None:
            self.timestamps = []
            self.values = []
            self.data_available = False
            self.logger.debug("Không có dữ liệu được cung cấp, detector được khởi tạo với dữ liệu trống")
        else:
            self.timestamps = np.array(timestamps)
            self.values = np.array(values)
            self.data_available = True
            self.logger.info(f"Detector được khởi tạo với {len(values)} điểm dữ liệu")
    
    def _format_timestamp(self, timestamp: int) -> str:
        # Chuyển đổi UNIX timestamp sang chuỗi thời gian đọc được.
        
        # Args:
        #     timestamp: UNIX timestamp
            
        # Returns:
        #     Chuỗi thời gian định dạng YYYY-MM-DD HH:MM:SS
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return str(timestamp)
    
    def cusum_detection(self, threshold=5.0, target=None):
        # Phát hiện bất thường bằng thuật toán CUSUM (Cumulative Sum).
        
        # Args:
        #     threshold: Ngưỡng phát hiện
        #     target: Giá trị mục tiêu (mặc định sẽ là giá trị trung bình của chuỗi)
            
        # Returns:
        #     Danh sách các chỉ mục bất thường
        if not self.data_available or len(self.values) < 2:
            self.logger.warning("Không đủ dữ liệu cho phát hiện CUSUM")
            return []
        
        # Sử dụng giá trị trung bình nếu không được chỉ định
        if target is None:
            target = np.mean(self.values)
        
        self.logger.debug(f"Thực hiện phát hiện CUSUM với ngưỡng={threshold} và target={target}")
        
        # Tính sai số (difference)
        diff = self.values - target
        
        # Tính tổng cộng dồn
        cusum_positive = np.zeros_like(diff)
        cusum_negative = np.zeros_like(diff)
        
        for i in range(1, len(diff)):
            cusum_positive[i] = max(0, cusum_positive[i-1] + diff[i])
            cusum_negative[i] = min(0, cusum_negative[i-1] + diff[i])
        
        # Tìm các điểm vượt ngưỡng
        anomalies = np.where((cusum_positive > threshold) | (cusum_negative < -threshold))[0]
        
        self.logger.info(f"Phát hiện CUSUM tìm thấy {len(anomalies)} bất thường với ngưỡng={threshold}")
        return anomalies.tolist()
    
    def zscore_detection(self, threshold=3.0):
        # Phát hiện bất thường bằng thuật toán Z-Score.
        
        # Args:
        #     threshold: Ngưỡng phát hiện (số lần độ lệch chuẩn)
            
        # Returns:
        #     Danh sách các chỉ mục bất thường
        if not self.data_available or len(self.values) < 2:
            self.logger.warning("Không đủ dữ liệu cho phát hiện Z-Score")
            return []
        
        self.logger.debug(f"Thực hiện phát hiện Z-Score với ngưỡng={threshold}")
        
        # Tính giá trị trung bình và độ lệch chuẩn
        mean = np.mean(self.values)
        std = np.std(self.values)
        
        # Tránh chia cho 0
        if std == 0:
            self.logger.warning("Độ lệch chuẩn bằng 0, không thể thực hiện phát hiện Z-Score")
            return []
        
        # Tính Z-Score và tìm các điểm vượt ngưỡng
        z_scores = np.abs((self.values - mean) / std)
        anomalies = np.where(z_scores > threshold)[0]
        
        self.logger.info(f"Phát hiện Z-Score tìm thấy {len(anomalies)} bất thường với ngưỡng={threshold}")
        return anomalies.tolist()
    
    def iqr_detection(self, k=1.5):
        # Phát hiện bất thường bằng phương pháp IQR (Interquartile Range).
        
        # Args:
        #     k: Hệ số ngưỡng (thường là 1.5 hoặc 3.0)
            
        # Returns:
        #     Danh sách các chỉ mục bất thường
        if not self.data_available or len(self.values) < 4:
            self.logger.warning("Không đủ dữ liệu cho phát hiện IQR")
            return []
        
        self.logger.debug(f"Thực hiện phát hiện IQR với k={k}")
        
        # Tính các tứ phân vị
        q1 = np.percentile(self.values, 25)
        q3 = np.percentile(self.values, 75)
        iqr = q3 - q1
        
        # Tránh chia cho 0
        if iqr == 0:
            self.logger.warning("IQR bằng 0, không thể thực hiện phát hiện IQR")
            return []
        
        # Tính biên trên và biên dưới
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        # Tìm các điểm nằm ngoài biên
        anomalies = np.where((self.values < lower_bound) | (self.values > upper_bound))[0]
        
        self.logger.info(f"Phát hiện IQR tìm thấy {len(anomalies)} bất thường với k={k}")
        return anomalies.tolist()
    
    def moving_average_detection(self, window=5, threshold=2.0):
        # Phát hiện bất thường bằng phương pháp Moving Average.
        
        # Args:
        #     window: Kích thước cửa sổ trượt
        #     threshold: Ngưỡng phát hiện (số lần độ lệch chuẩn)
            
        # Returns:
        #     Danh sách các chỉ mục bất thường
        if not self.data_available or len(self.values) <= window:
            self.logger.warning(f"Không đủ dữ liệu cho phát hiện Moving Average với window={window}")
            return []
        
        self.logger.debug(f"Thực hiện phát hiện Moving Average với window={window}, threshold={threshold}")
        
        anomalies = []
        
        for i in range(window - 1, len(self.values)):
            window_start = max(0, i - window + 1)
            window_end = i + 1
            window_values = self.values[window_start:window_end]
            window_mean = np.mean(window_values)
            window_std = np.std(window_values)
            
            # Tránh chia cho 0
            if window_std > 0 and abs(self.values[i] - window_mean) > threshold * window_std:
                anomalies.append(i)
        
        self.logger.info(f"Phát hiện Moving Average tìm thấy {len(anomalies)} bất thường với window={window}, threshold={threshold}")
        return anomalies
    
    def ensemble_detection(self, methods=None, min_votes=2):
        # Phát hiện bất thường bằng phương pháp kết hợp (ensemble).
        
        # Args:
        #     methods: Danh sách các phương pháp cần kết hợp
        #     min_votes: Số phiếu tối thiểu để xác định bất thường
            
        # Returns:
        #     Danh sách các chỉ mục bất thường
        if not self.data_available:
            self.logger.warning("Không có dữ liệu cho phát hiện Ensemble")
            return []
        
        if methods is None:
            methods = ['zscore', 'iqr', 'moving_average', 'cusum']
        
        self.logger.debug(f"Thực hiện phát hiện Ensemble với methods={methods}, min_votes={min_votes}")
        
        # Chạy các phương pháp phát hiện đã chọn
        all_anomalies = {}
        
        if 'zscore' in methods:
            all_anomalies['zscore'] = self.zscore_detection()
        
        if 'iqr' in methods:
            all_anomalies['iqr'] = self.iqr_detection()
        
        if 'moving_average' in methods:
            all_anomalies['moving_average'] = self.moving_average_detection()
        
        if 'cusum' in methods:
            all_anomalies['cusum'] = self.cusum_detection()
        
        # Đếm số phiếu cho mỗi chỉ số
        votes = {}
        for method, anomalies in all_anomalies.items():
            for idx in anomalies:
                votes[idx] = votes.get(idx, 0) + 1
        
        # Trả về các chỉ số có ít nhất min_votes
        ensemble_anomalies = [idx for idx, vote_count in votes.items() if vote_count >= min_votes]
        ensemble_anomalies.sort()
        
        self.logger.info(f"Phát hiện Ensemble tìm thấy {len(ensemble_anomalies)} bất thường với min_votes={min_votes}")
        return ensemble_anomalies
    
    def get_anomaly_points(self, anomaly_indices):
        if not self.data_available:
            self.logger.warning("Không có dữ liệu để lấy điểm bất thường")
            return []
        
        anomaly_points = []
        for idx in anomaly_indices:
            if 0 <= idx < len(self.timestamps):
                anomaly_points.append([int(self.timestamps[idx]), float(self.values[idx])])
        
        return anomaly_points
    
    def analyze(self, method='ensemble', **kwargs):
        # Phân tích dữ liệu và phát hiện bất thường.
        # Args:
        #     method: Phương pháp phát hiện ('zscore', 'iqr', 'moving_average', 'cusum', 'ensemble')
        #     **kwargs: Các tham số bổ sung cho phương pháp phát hiện

        # Returns:
        #     Dict chứa kết quả phân tích

        if not self.data_available:
            self.logger.warning("Không có dữ liệu cho phân tích")
            return {"error": "Không có dữ liệu cho phân tích"}
        
        self.logger.info(f"Phân tích dữ liệu với method={method} và parameters={kwargs}")
        
        # Chọn phương pháp phát hiện phù hợp
        try:
            if method == 'zscore':
                threshold = float(kwargs.get('threshold', 3.0))
                anomaly_indices = self.zscore_detection(threshold=threshold)
            elif method == 'iqr':
                k = float(kwargs.get('k', 1.5))
                anomaly_indices = self.iqr_detection(k=k)
            elif method == 'moving_average':
                window = int(kwargs.get('window', 5))
                threshold = float(kwargs.get('threshold', 2.0))
                anomaly_indices = self.moving_average_detection(window=window, threshold=threshold)
            elif method == 'cusum':
                threshold = float(kwargs.get('threshold', 5.0))
                target = kwargs.get('target')
                if target is not None:
                    target = float(target)
                anomaly_indices = self.cusum_detection(threshold=threshold, target=target)
            elif method == 'ensemble':
                methods = kwargs.get('methods', None)
                min_votes = int(kwargs.get('min_votes', 2))
                anomaly_indices = self.ensemble_detection(methods=methods, min_votes=min_votes)
            else:
                error_msg = f"Phương pháp phát hiện không xác định: {method}"
                self.logger.error(error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Lỗi khi thực hiện phát hiện bất thường với phương pháp {method}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"error": error_msg}
        
        # Lấy các điểm bất thường (timestamp, value pairs)
        anomaly_points = self.get_anomaly_points(anomaly_indices)
        
        # Định dạng lại các điểm bất thường để hiển thị thời gian dễ đọc
        formatted_anomaly_points = []
        for point in anomaly_points:
            formatted_point = {
                "timestamp": point[0],
                "value": point[1],
                "time": self._format_timestamp(point[0])
            }
            formatted_anomaly_points.append(formatted_point)
        
        # Tính tỷ lệ phần trăm bất thường
        anomaly_percentage = len(anomaly_indices) / len(self.values) * 100 if self.values.size > 0 else 0
        
        # Trả về kết quả phân tích
        result = {
            "method": method,
            "parameters": kwargs,
            "anomaly_indices": anomaly_indices,
            "anomaly_points": formatted_anomaly_points,
            "anomaly_count": len(anomaly_indices),
            "anomaly_percentage": round(anomaly_percentage, 2),
            "total_points": len(self.values),
            "summary": {
                "min": float(np.min(self.values)) if self.values.size > 0 else None,
                "max": float(np.max(self.values)) if self.values.size > 0 else None,
                "mean": float(np.mean(self.values)) if self.values.size > 0 else None,
                "std": float(np.std(self.values)) if self.values.size > 0 else None
            }
        }
        
        self.logger.info(f"Phân tích hoàn tất: tìm thấy {len(anomaly_indices)} bất thường "
                        f"({anomaly_percentage:.2f}% dữ liệu)")
        
        return result

def select_algorithm_by_anomaly_type(anomaly_type: AnomalyType) -> DetectionAlgorithm:
    # Tự động chọn thuật toán phù hợp dựa trên loại bất thường.
    # Args:
    #     anomaly_type: Loại bất thường cần phát hiện
    # Returns:
    #     Thuật toán phát hiện phù hợp
    if anomaly_type == AnomalyType.SPIKE:
        return DetectionAlgorithm.ZSCORE # MAD
    elif anomaly_type == AnomalyType.CHANGE_MEAN: # Known timestamp
        return DetectionAlgorithm.CUSUM 
    # Unknown timestamp
    elif anomaly_type in (AnomalyType.INCREASE, AnomalyType.DECREASE):
        return DetectionAlgorithm.MOVING_AVERAGE