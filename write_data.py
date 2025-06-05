import requests
import time
import random
from typing import Dict, Optional, List

# Configuration
VM_URL = "http://localhost:8428"  # or use 8480 if using vmcluster (vminsert)
WRITE_ENDPOINT = f"{VM_URL}/api/v1/import/prometheus"

def write_timeseries(
    metric_name: str, 
    labels: Dict[str, str], 
    value: float, 
    timestamp: Optional[float] = None
) -> None:
    if timestamp is None:
        timestamp = time.time()

    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    line = f'{metric_name}{{{label_str}}} {value} {timestamp}'
    
    try:
        response = requests.post(WRITE_ENDPOINT, data=line)
        response.raise_for_status()
        print(f"Successfully wrote data: {line}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to write data: {e}")


class AnomalyRange:
    def __init__(self, start: int, end: int, amplitude: float, anomaly_type: str = "spike"):
        self.start = start
        self.end = end
        self.amplitude = amplitude
        self.anomaly_type = anomaly_type  # "spike", "dip", "change_mean", "increase", "decrease"

class TimeSeriesGenerator:
    def __init__(
        self,
        length: int = 1000,
        base_value: float = 10.0,
        noise: float = 0.5,
        anomaly_ranges: Optional[List[AnomalyRange]] = None,
    ):
        self.length = length
        self.base_value = base_value
        self.noise = noise
        self.anomaly_ranges = anomaly_ranges or []

    def add_anomaly_range(self, start: int, end: int, amplitude: float, anomaly_type: str = "spike"):
        self.anomaly_ranges.append(AnomalyRange(start, end, amplitude, anomaly_type))

    def generate(self) -> List[float]:
        data = []
        mean_shift = 0.0  # for "change_mean" anomaly
        for i in range(self.length):
            base = self.base_value + mean_shift
            value = base + random.uniform(-self.noise, self.noise)

            for anomaly in self.anomaly_ranges:
                if anomaly.anomaly_type == "change_mean" and i == anomaly.start:
                    mean_shift += anomaly.amplitude

                if anomaly.start <= i < anomaly.end:
                    if anomaly.anomaly_type == "spike":
                        value += anomaly.amplitude
                    elif anomaly.anomaly_type == "dip":
                        value -= anomaly.amplitude
                    elif anomaly.anomaly_type == "increase":
                        progress = (i - anomaly.start) / (anomaly.end - anomaly.start)
                        value += anomaly.amplitude * progress
                    elif anomaly.anomaly_type == "decrease":
                        progress = (i - anomaly.start) / (anomaly.end - anomaly.start)
                        value -= anomaly.amplitude * progress

            data.append(value)
        return data


if __name__ == "__main__":
    gen1 = TimeSeriesGenerator(length=200, base_value=10.0, noise=1)
    gen1.add_anomaly_range(50, 51, 20, "spike")
    # gen1.add_anomaly_range(120, 125, 15, "dip")
    series1 = gen1.generate()

    # gen2 = TimeSeriesGenerator(length=200, base_value=10.0, noise=1)
    # gen2.add_anomaly_range(60, 80, 8, "increase")
    # gen2.add_anomaly_range(90, 110, 6, "decrease")
    # series2 = gen2.generate()

    # gen3 = TimeSeriesGenerator(length=200, base_value=10.0, noise=1)
    # gen3.add_anomaly_range(150, 151, 5, "change_mean")
    # series3 = gen3.generate()

    # Ghi dữ liệu có thời gian đồng bộ
    start_time = time.time()
    for i, value in enumerate(series1):
        write_timeseries("spike_dip_metric_7", {"host": "localhost", "anomaly": "spike_dip"}, value, start_time + i * 30)

    # for i, value in enumerate(series2):
    #     write_timeseries("increase_decrease_metric", {"host": "localhost", "anomaly": "trend"}, value, start_time + i * 5)

    # for i, value in enumerate(series3):
    #     write_timeseries("change_mean_metric", {"host": "localhost", "anomaly": "mean_shift"}, value, start_time + i * 5)
