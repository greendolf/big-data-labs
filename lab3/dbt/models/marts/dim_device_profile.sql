/* Витрина профиля устройства для сценария Fraud Detection:
   риск = (1-w) * доля атак + w * нормированная средняя тяжесть,
   w = dbt-переменная risk_severity_weight (по умолчанию 0.3; канарейка меняет её). */
{% set w = var('risk_severity_weight', 0.3) | float %}
with per_device as (
    select
        device_name,
        count(*) as events,
        sum(case when is_attack then 1 else 0 end) as attacks,
        avg(case when is_attack then hh_magnitude end) as avg_attack_magnitude,
        max(severity) as max_severity
    from {{ ref('stg_events') }}
    group by device_name
), dominant_subtype as (
    select device_name, attack_subtype as top_subtype
    from (
        select device_name, attack_subtype,
               row_number() over (partition by device_name order by count(*) desc) as rn
        from {{ ref('stg_events') }}
        where is_attack
        group by device_name, attack_subtype
    )
    where rn = 1
)
select
    p.device_name,
    p.events,
    p.attacks,
    p.attacks * 1.0 / p.events as attack_rate,
    d.top_subtype,
    p.avg_attack_magnitude,
    p.max_severity,
    round((1 - {{ w }}) * (p.attacks * 1.0 / p.events)
          + {{ w }} * (p.max_severity / 3.0), 4) as risk_score,
    now() as profiled_at
from per_device p
left join dominant_subtype d using (device_name)
