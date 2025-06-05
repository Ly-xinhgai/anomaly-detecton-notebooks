anomaly-detection-system/
├── config/
│   ├── config.yml           # File cấu hình hệ thống
│   └── logging.yml          # Cấu hình logging
├── app/
│   ├── __init__.py          # Package init
│   ├── main.py              # File chính để khởi động API server
│   ├── api/
│   │   ├── __init__.py      # Package init
│   │   ├── routes.py        # Định nghĩa các route API
│   │   └── endpoints/
│   │       ├── __init__.py  # Package init
│   │       ├── metrics.py   # Endpoints xử lý yêu cầu metric
│   │       └── anomalies.py # Endpoints phát hiện bất thường
│   ├── services/
│   │   ├── __init__.py      # Package init
│   │   ├── victoria_metrics.py   # Service giao tiếp với VictoriaMetrics
│   │   └── anomaly_detection.py  # Service phát hiện bất thường
│   ├── core/
│   │   ├── __init__.py      # Package init
│   │   ├── config.py        # Xử lý cấu hình
│   │   ├── logger.py        # Setup logging
│   │   └── exceptions.py    # Custom exceptions
│   ├── models/
│   │   ├── __init__.py      # Package init
│   │   ├── threshold.py     # Model cho ngưỡng phát hiện
│   │   └── anomaly.py       # Model cho dữ liệu bất thường
│   └── utils/
│       ├── __init__.py      # Package init
│       ├── time_utils.py    # Các hàm tiện ích xử lý thời gian
│       └── statistics.py    # Các hàm tiện ích xử lý thống kê
├── tests/
│   ├── __init__.py          # Package init
│   ├── test_victoria_metrics.py # Unit test cho Victoria Metrics service
│   └── test_anomaly_detection.py # Unit test cho anomaly detection
├── .env                     # Biến môi trường
├── requirements.txt         # Thư viện dependencies
├── Dockerfile               # Để chạy trong container
└── README.md                # Hướng dẫn sử dụng


# Metric Query Server

Server HTTP đơn giản để truy vấn dữ liệu metric từ VictoriaMetrics dựa trên tên metric, thời gian bắt đầu, và thời gian kết thúc.

## Tính năng

- API endpoint để truy vấn dữ liệu metric từ VictoriaMetrics
- Hỗ trợ tham số thời gian bắt đầu, thời gian kết thúc, và bước thời gian
- Cung cấp API để lấy danh sách tất cả các metric có sẵn
- Tích hợp với FastAPI để cung cấp tài liệu API tự động

## Yêu cầu

- Python 3.8+
- FastAPI
- Uvicorn
- Requests
- Python-dateutil (tùy chọn, để hỗ trợ nhiều định dạng thời gian)
- Kết nối đến VictoriaMetrics

## Cài đặt

1. Clone repository:

```bash
git clone https://github.com/your-username/metric-query-server.git
cd metric-query-server
```

2. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

3. Chỉnh sửa URL VictoriaMetrics trong file `server.py` nếu cần:

```python
VICTORIA_METRICS_URL = "http://localhost:8428"  # Thay đổi URL này nếu cần
```

## Chạy Server

### Chạy trực tiếp

```bash
python server.py
```

Hoặc:

```bash
uvicorn server:app --reload
```

### Sử dụng Docker

1. Xây dựng image:

```bash
docker build -t metric-query-server .
```

2. Chạy container:

```bash
docker run -p 8000:8000 -e VICTORIA_METRICS_URL=http://your-vm-host:8428 metric-query-server
```

3. Sử dụng Docker Compose:

```bash
docker-compose up -d
```

## Sử dụng API

### Truy vấn Metric

```
GET /api/query?metric_name=<metric_name>&start_time=<start_time>&end_time=<end_time>&step=<step>
```

Tham số:
- `metric_name`: Tên metric hoặc truy vấn PromQL
- `start_time`: Thời gian bắt đầu (ISO format hoặc Unix timestamp)
- `end_time`: Thời gian kết thúc (ISO format hoặc Unix timestamp)
- `step`: Bước thời gian (ví dụ: 15s, 1m, 5m, 1h), mặc định là 15s

Ví dụ:
```
GET /api/query?metric_name=node_cpu_seconds_total&start_time=2023-04-01T00:00:00Z&end_time=2023-04-01T01:00:00Z&step=1m
```

### Lấy danh sách Metric

```
GET /api/metrics/list
```

## Client Example

Có thể sử dụng script `client.py` đi kèm để truy vấn dữ liệu từ command line:

### Truy vấn Metric

```bash
python client.py query "node_cpu_seconds_total" --start "2023-04-01T00:00:00Z" --end "2023-04-01T01:00:00Z" --step "1m"
```

### Lấy danh sách Metric

```bash
python client.py list
```

## Tài liệu API

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Tích hợp với VictoriaMetrics

Server này được thiết kế để tương tác với VictoriaMetrics thông qua API `/api/v1/query_range`. Đảm bảo rằng instance VictoriaMetrics của bạn đang chạy và có thể truy cập từ server này.

## License

MIT