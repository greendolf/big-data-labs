{{ config(materialized='incremental', unique_key='idempotency_key') }}

/* Факт злонамеренных событий: только атаки, идемпотентно по ключу. */
select
    idempotency_key,
    device_name,
    attack_type,
    attack_subtype,
    severity,
    event_time,
    is_late,
    batch_id,
    hh_magnitude,
    mi_dir_mean
from {{ ref('stg_events') }}
where is_attack
{% if is_incremental() %}
    and idempotency_key not in (select idempotency_key from {{ this }})
{% endif %}
