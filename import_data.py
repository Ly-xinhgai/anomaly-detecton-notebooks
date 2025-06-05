import requests
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Cấu hình logging
logger = logging.getLogger("import-data")

class BaseMetricsClient:
    # """Base class cho các client metrics."""
    
    def __init__(self, base_url: str, timeout: int = 30):

        # Khởi tạo client metrics.
        
        # Args:
        #     base_url: URL cơ sở của service metrics
        #     timeout: Thời gian chờ tối đa cho các yêu cầu HTTP (giây)
       
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def query_range(self, query: str, start: int, end: int, step: str = "1m") -> Dict[str, Any]:
        # Truy vấn dữ liệu từ service metrics trong khoảng thời gian.
        
        # Args:
        #     query: Truy vấn PromQL/MetricsQL
        #     start: Thời gian bắt đầu (UNIX timestamp)
        #     end: Thời gian kết thúc (UNIX timestamp)
        #     step: Bước thời gian
            
        # Returns:
        #     Dict chứa kết quả từ service metrics
        raise NotImplementedError("Phương thức này cần được triển khai ở lớp con")
    
    def extract_values_in_range(self, data: Dict[str, Any]) -> List[List]:
        # Trích xuất các giá trị từ kết quả query.
        
        # Args:
        #     data: Kết quả từ phương thức query_range
            
        # Returns:
        #     List của các cặp [timestamp, value]
    
        raise NotImplementedError("Phương thức này cần được triển khai ở lớp con")
    
    def get_timeseries_data(self, query: str, start: int, end: int, step: str = "1m") -> Tuple[List[int], List[float]]:
        # Lấy dữ liệu chuỗi thời gian từ service metrics.
        
        # Args:
        #     query: Truy vấn PromQL/MetricsQL
        #     start: Thời gian bắt đầu (UNIX timestamp)
        #     end: Thời gian kết thúc (UNIX timestamp)
        #     step: Bước thời gian
            
        # Returns:
        #     Tuple của hai list: (timestamps, values)
        # Truy vấn dữ liệu
        result = self.query_range(query, start, end, step)
        
        # Trích xuất các cặp [timestamp, value]
        point_pairs = self.extract_values_in_range(result)
        
        # Tách thành danh sách timestamps và values riêng biệt
        timestamps = []
        values = []
        
        if point_pairs:
            timestamps = [pair[0] for pair in point_pairs]
            values = [pair[1] for pair in point_pairs]
        
        return timestamps, values


class VictoriaMetricsClient(BaseMetricsClient):
    
    def query_range(self, query: str, start: int, end: int, step: str = "1m") -> Dict[str, Any]:

        # Truy vấn dữ liệu từ VictoriaMetrics trong khoảng thời gian.
        
        # Args:
        #     query: Truy vấn PromQL/MetricsQL
        #     start: Thời gian bắt đầu (UNIX timestamp)
        #     end: Thời gian kết thúc (UNIX timestamp)
        #     step: Bước thời gian (mặc định: 1m)
            
        # Returns:
        #     Dict chứa kết quả từ VictoriaMetrics

        try:
            url = f"{self.base_url}/api/v1/query_range"
            params = {
                'query': query,
                'start': start,
                'end': end,
                'step': step
            }
            
            self.logger.debug(f"Truy vấn VictoriaMetrics: {url} với tham số {params}")
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()  # Phát sinh lỗi nếu status không thành công
            
            result = response.json()
            self.logger.info(f"Đã nhận được kết quả từ VictoriaMetrics, status: {response.status_code}")
            
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Lỗi khi truy vấn VictoriaMetrics: {str(e)}")
            raise
    
    def extract_values_in_range(self, data: Dict[str, Any]) -> List[List]:
        # Trích xuất các giá trị từ kết quả VictoriaMetrics.
        
        # Args:
        #     data: Kết quả từ phương thức query_range
            
        # Returns:
        #     List của các cặp [timestamp, value]
        values = []
        try:
            if 'data' in data and 'result' in data['data']:
                result_count = len(data['data']['result'])
                self.logger.debug(f"Xử lý {result_count} chuỗi kết quả từ VictoriaMetrics")
                
                for result in data['data']['result']:
                    if 'values' in result:
                        for value in result['values']:
                            try:
                                timestamp = int(value[0])  # Chuyển timestamp sang kiểu int
                                metric_value = float(value[1])  # Chuyển giá trị sang float để xử lý cả giá trị thập phân
                                values.append([timestamp, metric_value])
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"Lỗi khi phân tích giá trị {value}: {str(e)}")
                                continue
            
            self.logger.info(f"Đã trích xuất {len(values)} điểm dữ liệu")
            return values
        except Exception as e:
            self.logger.error(f"Lỗi khi trích xuất giá trị: {str(e)}")
            return []
    
    def get_metrics_metadata(self) -> List[Dict[str, Any]]:
        # Lấy metadata về các metrics có sẵn.
        
        # Returns:
        #     Danh sách các metadata của metrics
        try:
            url = f"{self.base_url}/api/v1/label/__name__/values"
            
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            if 'data' in result:
                return result['data']
            return []
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy metadata metrics: {str(e)}")
            return []


class PrometheusClient(BaseMetricsClient):
    
    def query_range(self, query: str, start: int, end: int, step: str = "15s") -> Dict[str, Any]:
        # Truy vấn dữ liệu dựa trên thời gian từ Prometheus.
        
        # Args:
        #     query: Truy vấn PromQL
        #     start: Thời gian bắt đầu (UNIX timestamp)
        #     end: Thời gian kết thúc (UNIX timestamp)
        #     step: Bước thời gian (mặc định: 15s)
            
        # Returns:
        #     Dict chứa kết quả từ Prometheus
        try:
            url = f"{self.base_url}/api/v1/query_range"
            params = {
                "query": query,
                "start": start,
                "end": end,
                "step": step
            }
            
            self.logger.debug(f"Truy vấn Prometheus: {url} với tham số {params}")
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()  # Phát sinh lỗi nếu status không thành công
            
            result = response.json()
            self.logger.info(f"Đã nhận được kết quả từ Prometheus, status: {response.status_code}")
            
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Lỗi khi truy vấn Prometheus: {str(e)}")
            raise
    
    def extract_values_in_range(self, data: Dict[str, Any]) -> List[List]:
        # Trích xuất chuỗi giá trị timestamp và values từ kết quả Prometheus.
        
        # Args:
        #     data: Kết quả từ phương thức query_range
            
        # Returns:
        #     List của các cặp [timestamp, value]
    
        values = []
        try:
            if 'data' in data and 'result' in data['data']:
                result_count = len(data['data']['result'])
                self.logger.debug(f"Xử lý {result_count} chuỗi kết quả từ Prometheus")
                
                for series in data['data']['result']:
                    if 'values' in series:
                        for point in series['values']:
                            try:
                                timestamp = int(point[0])
                                value = float(point[1])
                                values.append([timestamp, value])
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"Lỗi khi phân tích giá trị {point}: {str(e)}")
                                continue
            
            self.logger.info(f"Đã trích xuất {len(values)} điểm dữ liệu")
            return values
        except Exception as e:
            self.logger.error(f"Lỗi khi trích xuất giá trị: {str(e)}")
            return []
    
    def query(self, query: str) -> Dict[str, Any]:
        # Truy vấn instant query từ Prometheus.
        
        # Args:
        #     query: Truy vấn PromQL
            
        # Returns:
        #     Dict chứa kết quả từ Prometheus
        try:
            url = f"{self.base_url}/api/v1/query"
            params = {"query": query}
            
            self.logger.debug(f"Truy vấn instant Prometheus: {url} với tham số {params}")
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            self.logger.info(f"Đã nhận được kết quả từ instant query Prometheus, status: {response.status_code}")
            
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Lỗi khi truy vấn instant Prometheus: {str(e)}")
            raise


# Factory cho client metrics
def create_metrics_client(client_type: str, base_url: str, timeout: int = 30) -> BaseMetricsClient:
    
    # Tạo client metrics dựa trên loại được chỉ định.
    
    # Args:
    #     client_type: Loại client ("prometheus" hoặc "victoriametrics")
    #     base_url: URL cơ sở của service
    #     timeout: Thời gian chờ tối đa cho các yêu cầu HTTP (giây)
        
    # Returns:
    #     Một instance của client metrics
    if client_type.lower() == "prometheus":
        return PrometheusClient(base_url, timeout)
    elif client_type.lower() == "victoriametrics":
        return VictoriaMetricsClient(base_url, timeout)
    else:
        raise ValueError(f"Loại client không được hỗ trợ: {client_type}")