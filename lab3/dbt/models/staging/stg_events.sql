{{ config(
    materialized='incremental',
    unique_key='idempotency_key',
    schema='staging'
) }}

/* Staging: нормализация имён, типизация, бизнес-признаки. Инкрементальна по
   idempotency_key — повторная доставка существующего ключа обновляет строку,
   а не дублирует (ELT-идемпотентность на уровне модели). */
with src as (
    select * from {{ source('lake', 'events') }}
    {% if is_incremental() %}
        -- новые ключи (включая поздно пришедшие) добавляются; повторные ключи
        -- уже в модели — анти-джойн делает модель идемпотентной
        where idempotency_key not in (select idempotency_key from {{ this }})
    {% endif %}
)
select
    idempotency_key,
    batch_id,
    event_time,
    ingested_at,
    now() as _loaded_at,
    is_late,
    "Device_Name"    as device_name,
    "Attack"         as attack_type,
    "Attack_subType" as attack_subtype,
    cast(label as integer) as label,
    case when "Attack" = 'Normal' then false else true end as is_attack,
    {{ attack_severity('"Attack"', '"Attack_subType"') }} as severity,
    "MI_dir_L0.1_mean"     as mi_dir_mean,
    "MI_dir_L0.1_variance" as mi_dir_variance,
    "H_L0.1_mean"          as h_mean,
    "H_L0.1_variance"      as h_variance,
    "HH_L0.1_magnitude"    as hh_magnitude,
    "HH_L0.1_mean"         as hh_mean
from src
