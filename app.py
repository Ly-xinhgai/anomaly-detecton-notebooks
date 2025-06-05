from fastapi import FastAPI, HTTPException, Query, Depends
import uvicorn
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import logging
import os
from dotenv import load_dotenv
import json
# Import các module tự tạo
from import_data import create_metrics_client, BaseMetricsClient
from algorithms import (
    z_score, modified_z_score, moving_z_score,
    cusum, page_hinkley, ewma_control_chart, detect_mean_shift_at_known_points,
    detect_increasing_decreasing_sequences, mann_kendall_trend_test, linear_regression_slope_test,
    ensemble_voting, calculate_anomaly_scores, evaluate_anomaly_detection
)

# Tải biến môi trường
load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anomaly-detection-server")

# Khởi tạo FastAPI app
app = FastAPI(title="Anomaly Detection Server - VictoriaMetrics")

# Lấy URL từ .env
VICTORIA_METRICS_URL = os.getenv('VICTORIA_METRICS_URL')

if not VICTORIA_METRICS_URL:
    logger.error("VICTORIA_METRICS_URL không được cấu hình trong file .env")
    raise ValueError("VICTORIA_METRICS_URL phải được cấu hình")

# Định nghĩa enum cho các loại bất thường
class AnomalyType(str, Enum):
    SPIKE = "spike"
    CHANGE_MEAN = "change_mean"
    INCREASE = "increase"
    DECREASE = "decrease"
    ALL = "all"

# Định nghĩa enum cho các thuật toán
class DetectionAlgorithm(str, Enum):
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    MOVING_ZSCORE = "moving_zscore"
    CUSUM = "cusum"
    PAGE_HINKLEY = "page_hinkley"
    EWMA = "ewma"
    MANN_KENDALL = "mann_kendall"
    LINEAR_REGRESSION = "linear_regression"
    ENSEMBLE = "ensemble"

def get_metrics_client() -> BaseMetricsClient:
    """
    Dependency để lấy VictoriaMetrics client.
    """
    return create_metrics_client("victoriametrics", VICTORIA_METRICS_URL)

def convert_to_timestamp(time_str: str) -> int:
    """
    Chuyển đổi thời gian từ chuỗi sang UNIX timestamp.
    """
    try:
        datetime_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        return int(datetime_obj.timestamp())
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Định dạng thời gian không hợp lệ. Sử dụng YYYY-MM-DD HH:MM:SS. Lỗi: {str(e)}"
        )

def select_algorithm_by_anomaly_type(anomaly_type: AnomalyType) -> DetectionAlgorithm:
    """
    Tự động chọn thuật toán phù hợp dựa trên loại bất thường.
    """
    if anomaly_type == AnomalyType.SPIKE:
        return DetectionAlgorithm.ZSCORE
    elif anomaly_type == AnomalyType.CHANGE_MEAN:
        return DetectionAlgorithm.CUSUM
    elif anomaly_type in (AnomalyType.INCREASE, AnomalyType.DECREASE):
        return DetectionAlgorithm.MANN_KENDALL
    else:  # AnomalyType.ALL
        return DetectionAlgorithm.ENSEMBLE

def apply_anomaly_detection(values: List[float], timestamps: List[int], 
                          algorithm: DetectionAlgorithm, **params) -> Dict[str, Any]:
    """
    Áp dụng thuật toán phát hiện bất thường và trả về kết quả chuẩn hóa.
    """
    anomaly_indices = []
    anomaly_details = {}
    
    try:
        if algorithm == DetectionAlgorithm.ZSCORE:
            threshold = params.get('threshold', 3.0)
            anomaly_mask = z_score(values, threshold)
            anomaly_indices = [i for i, is_anomaly in enumerate(anomaly_mask) if is_anomaly]
            anomaly_details = {'threshold': threshold, 'method': 'Z-Score'}
            
        elif algorithm == DetectionAlgorithm.MODIFIED_ZSCORE:
            threshold = params.get('threshold', 3.5)
            anomaly_mask = modified_z_score(values, threshold)
            anomaly_indices = [i for i, is_anomaly in enumerate(anomaly_mask) if is_anomaly]
            anomaly_details = {'threshold': threshold, 'method': 'Modified Z-Score'}
            
        elif algorithm == DetectionAlgorithm.MOVING_ZSCORE:
            window = params.get('window', 30)
            threshold = params.get('threshold', 3.0)
            anomaly_mask = moving_z_score(values, window, threshold)
            anomaly_indices = [i for i, is_anomaly in enumerate(anomaly_mask) if is_anomaly]
            anomaly_details = {'window': window, 'threshold': threshold, 'method': 'Moving Z-Score'}
            
        elif algorithm == DetectionAlgorithm.CUSUM:
            threshold = params.get('threshold', 5.0)
            drift = params.get('drift', 0.01)
            change_points = cusum(values, threshold, drift)
            anomaly_indices = [cp[0] for cp in change_points]
            anomaly_details = {'threshold': threshold, 'drift': drift, 'method': 'CUSUM'}
            
        elif algorithm == DetectionAlgorithm.PAGE_HINKLEY:
            delta = params.get('delta', 1.0)
            threshold = params.get('threshold', 10.0)
            anomaly_indices = page_hinkley(values, delta, threshold)
            anomaly_details = {'delta': delta, 'threshold': threshold, 'method': 'Page-Hinkley'}
            
        elif algorithm == DetectionAlgorithm.EWMA:
            lambda_param = params.get('lambda', 0.2)
            threshold_factor = params.get('threshold_factor', 3.0)
            anomaly_mask = ewma_control_chart(values, lambda_param, threshold_factor)
            anomaly_indices = [i for i, is_anomaly in enumerate(anomaly_mask) if is_anomaly]
            anomaly_details = {'lambda': lambda_param, 'threshold_factor': threshold_factor, 'method': 'EWMA'}
            
        elif algorithm == DetectionAlgorithm.MANN_KENDALL:
            alpha = params.get('alpha', 0.05)
            result = mann_kendall_trend_test(values, alpha)
            # Mann-Kendall trả về xu hướng tổng thể, không phải anomaly points cụ thể
            anomaly_indices = []  # Có thể customize logic này
            anomaly_details = result
            anomaly_details['method'] = 'Mann-Kendall'
            
        elif algorithm == DetectionAlgorithm.LINEAR_REGRESSION:
            window = params.get('window', 30)
            threshold = params.get('threshold', 0.1)
            slope_results = linear_regression_slope_test(values, window, threshold)
            anomaly_indices = [result[0] for result in slope_results if result[2]]  # significant slopes
            anomaly_details = {'window': window, 'threshold': threshold, 'method': 'Linear Regression'}
            
        elif algorithm == DetectionAlgorithm.ENSEMBLE:
            methods = params.get('methods', ['z_score', 'modified_z_score'])
            voting_threshold = params.get('voting_threshold', 0.5)
            # Ensemble cần được implement với các method có sẵn
            anomaly_mask = ensemble_voting(values, methods, voting_threshold)
            anomaly_indices = [i for i, is_anomaly in enumerate(anomaly_mask) if is_anomaly]
            anomaly_details = {'methods': methods, 'voting_threshold': voting_threshold, 'method': 'Ensemble'}
            
        else:
            raise ValueError(f"Thuật toán không được hỗ trợ: {algorithm}")
            
    except Exception as e:
        logger.error(f"Lỗi khi áp dụng thuật toán {algorithm}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi áp dụng thuật toán: {str(e)}")
    
    # Tạo anomaly points với timestamp và value
    anomaly_points = []
    for idx in anomaly_indices:
        if 0 <= idx < len(timestamps):
            anomaly_points.append({
                "index": idx,
                "timestamp": timestamps[idx],
                "value": values[idx],
                "time": datetime.fromtimestamp(timestamps[idx]).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Tính toán thống kê
    anomaly_percentage = len(anomaly_indices) / len(values) * 100 if values else 0
    
    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_points": anomaly_points,
        "anomaly_count": len(anomaly_indices),
        "anomaly_percentage": round(anomaly_percentage, 2),
        "total_points": len(values),
        "algorithm_details": anomaly_details,
        "summary": {
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None,
            "mean": float(sum(values) / len(values)) if values else None,
            "std": float((sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5) if values else None
        }
    }

@app.post("/api/anomaly-detection")
async def detect_anomalies(
    query: str = Query(..., description="Truy vấn MetricsQL"),
    start_time: float = Query(..., description="Thời gian bắt đầu (YYYY-MM-DD HH:MM:SS)"),
    end_time: float = Query(..., description="Thời gian kết thúc (YYYY-MM-DD HH:MM:SS)"),
    anomaly_type: AnomalyType = Query(AnomalyType.SPIKE, description="Loại bất thường cần phát hiện"),
    algorithm: Optional[DetectionAlgorithm] = Query(None, description="Thuật toán phát hiện (tự động chọn nếu không chỉ định)"),
    threshold: Optional[float] = Query(None, description="Ngưỡng phát hiện (tùy thuộc thuật toán)"),
    window: Optional[int] = Query(None, description="Kích thước cửa sổ (cho một số thuật toán)"),
    metrics_client: BaseMetricsClient = Depends(get_metrics_client)
):
    """
    API endpoint phát hiện bất thường trong dữ liệu chuỗi thời gian từ VictoriaMetrics.
    """
    try:
        # Tự động chọn thuật toán nếu không được chỉ định
        if algorithm is None:
            algorithm = select_algorithm_by_anomaly_type(anomaly_type)
        
        logger.info(f"Phát hiện bất thường cho metric '{query}' với loại '{anomaly_type}' "
                   f"và thuật toán '{algorithm}' từ VictoriaMetrics")
        
        # Chuyển đổi thời gian từ chuỗi sang timestamp
        start_timestamp = int(start_time)
        end_timestamp = int(end_time)
        
        # Kiểm tra thời gian hợp lệ
        if start_timestamp >= end_timestamp:
            raise HTTPException(
                status_code=400, 
                detail="Thời gian bắt đầu phải nhỏ hơn thời gian kết thúc"
            )
        
        # Truy vấn dữ liệu từ VictoriaMetrics
        result = metrics_client.query_range(query, start_timestamp, end_timestamp)
        
        # Kiểm tra kết quả
        if not result or 'data' not in result or 'result' not in result['data']:
            raise HTTPException(status_code=404, detail="Không có dữ liệu hoặc có lỗi trong truy vấn")
        
        # Trích xuất dữ liệu thời gian
        values_data = metrics_client.extract_values_in_range(result)
        
        if not values_data:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu hợp lệ trong kết quả")
        
        # Tách timestamp và values
        timestamps = [int(point[0]) for point in values_data]
        values = [float(point[1]) for point in values_data]
        
        # Chuẩn bị tham số cho thuật toán
        detection_params = {}
        if threshold is not None:
            detection_params['threshold'] = threshold
        if window is not None:
            detection_params['window'] = window
            
        # Thiết lập tham số mặc định dựa trên loại bất thường
        if anomaly_type == AnomalyType.SPIKE and algorithm == DetectionAlgorithm.ZSCORE:
            detection_params.setdefault('threshold', 3.0)
        elif anomaly_type == AnomalyType.CHANGE_MEAN and algorithm == DetectionAlgorithm.CUSUM:
            detection_params.setdefault('threshold', 5.0)
            detection_params.setdefault('drift', 0.01)
        
        # Thực hiện phát hiện bất thường
        analysis_result = apply_anomaly_detection(values, timestamps, algorithm, **detection_params)
        
        # Thêm thông tin về truy vấn vào kết quả
        analysis_result.update({
            "query": query,
            "start_time": start_time,
            "end_time": end_time,
            "anomaly_type": anomaly_type,
            "data_source": "victoriametrics",
            "algorithm_used": algorithm,
            "parameters_used": detection_params
        })
        
        return analysis_result
    
    except HTTPException:
        # Ném lại HTTPException để không bị wrap
        raise
    except Exception as e:
        logger.error(f"Lỗi khi phát hiện bất thường: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi phát hiện bất thường: {str(e)}")

def get_anomaly_description(anomaly_type: AnomalyType) -> str:
    """
    Trả về mô tả cho loại bất thường.
    """
    descriptions = { 
        AnomalyType.SPIKE: "Phát hiện đỉnh bất thường (spike) trong dữ liệu",
        AnomalyType.CHANGE_MEAN: "Phát hiện thay đổi đột ngột trong giá trị trung bình",
        AnomalyType.INCREASE: "Phát hiện xu hướng tăng bất thường",
        AnomalyType.DECREASE: "Phát hiện xu hướng giảm bất thường",
        AnomalyType.ALL: "Kết hợp tất cả các loại phát hiện bất thường"
    }
    return descriptions.get(anomaly_type, "Loại bất thường không xác định")

def get_algorithm_description(algorithm: DetectionAlgorithm) -> str:
    """
    Trả về mô tả cho thuật toán phát hiện.
    """
    descriptions = {
        DetectionAlgorithm.ZSCORE: "Phát hiện dựa trên số lần độ lệch chuẩn từ giá trị trung bình",
        DetectionAlgorithm.MODIFIED_ZSCORE: "Z-Score cải tiến sử dụng MAD, robust hơn với outliers",
        DetectionAlgorithm.MOVING_ZSCORE: "Z-Score với cửa sổ trượt cho phát hiện online",
        DetectionAlgorithm.CUSUM: "Phát hiện thay đổi dựa trên tổng tích lũy (Cumulative Sum)",
        DetectionAlgorithm.PAGE_HINKLEY: "Sequential change point detection",
        DetectionAlgorithm.EWMA: "Exponentially Weighted Moving Average control chart",
        DetectionAlgorithm.MANN_KENDALL: "Non-parametric trend test",
        DetectionAlgorithm.LINEAR_REGRESSION: "Phát hiện trend bằng slope analysis",
        DetectionAlgorithm.ENSEMBLE: "Kết hợp nhiều thuật toán phát hiện để tăng độ chính xác"
    }
    return descriptions.get(algorithm, "Thuật toán không xác định")

@app.get("/")
async def root():
    """
    API endpoint mặc định, trả về thông tin tổng quan về server.
    """
    return {
        "message": "Anomaly Detection Server đang chạy với VictoriaMetrics",
        "data_source": "VictoriaMetrics",
        "api_endpoints": {
            "anomaly_detection": "/api/anomaly-detection",
        },
        "version": "3.0",
        "available_algorithms": len(DetectionAlgorithm),
        "victoria_metrics_url": VICTORIA_METRICS_URL
    }

# Khởi động ứng dụng
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Khởi động server trên port {port} với VictoriaMetrics: {VICTORIA_METRICS_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)