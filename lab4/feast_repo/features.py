"""Feast-определения фичей ЛР4 (проект lab4_fraud).

Entity device + FeatureView почасовых агрегатов активности устройства.
Источник — parquet, который генерирует lab4.features.build_feature_table().
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType

device = Entity(name="device", join_keys=["Device_Name"],
                value_type=ValueType.STRING,
                description="Устройство IoT (аналог клиента в fraud-сценарии)")

device_stats_source = FileSource(
    path="device_hourly_stats.parquet",   # относительно корня feature repo
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

device_stats = FeatureView(
    name="device_stats",
    entities=[device],
    ttl=timedelta(days=2),
    schema=[
        Field(name="event_count_1h", dtype=Float32),
        Field(name="avg_magnitude_1h", dtype=Float32),
        Field(name="avg_radius_1h", dtype=Float32),
        Field(name="avg_mi_mean_1h", dtype=Float32),
        Field(name="std_mi_mean_1h", dtype=Float32),
        Field(name="avg_weight_1h", dtype=Float32),
        Field(name="p95_mag_1h", dtype=Float32),
    ],
    online=True,
    source=device_stats_source,
    tags={"layer": "online+offline"},
)

DEVICE_STATS_REFS = [f"device_stats:{f.name}" for f in device_stats.schema]
