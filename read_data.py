from prometheus_api_client import PrometheusConnect
from datetime import datetime
from typing import Any, Dict, List
import json

# Configuration
VM_URL = "http://localhost:8428"

def read_timeseries_range(
    query: str, 
    start_time: datetime, 
    end_time: datetime, 
    step: str = "1m"
) -> List[Dict[str, Any]]:
    prom = PrometheusConnect(url=VM_URL, disable_ssl=True)
    return prom.custom_query_range(
        query=query,
        start_time=start_time,
        end_time=end_time,
        step=step
    )

def convert_timestamp_to_datetime(timestamp: str) -> datetime:
    return datetime.fromtimestamp(float(timestamp))

if __name__ == "__main__":
    query = 'spike_dip_metric'
    start_time = convert_timestamp_to_datetime("1747378011.6439998")
    end_time = convert_timestamp_to_datetime("1747380479.225")
    step = "5s"

    data = read_timeseries_range(query, start_time, end_time, step)
    value_array: list[float] = []
    for result in data:
        value_array.extend([float(value) for _, value in result.get("values", [])])

    with open("values_array.json", "w") as f:
        json.dump(value_array, f, indent=4)
    print("Mảng giá trị:")
    print(value_array)