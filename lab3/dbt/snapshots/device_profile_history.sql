{% snapshot device_profile_history %}

{{ config(
    target_schema='history',
    unique_key='device_name',
    strategy='check',
    check_cols=['events', 'attacks', 'attack_rate', 'risk_score']
) }}

/* SCD2-история профилей: каждый прогон dbt фиксирует версии витрины.
   Это «Time Travel» уровня витрины: запрос WHERE '2026-11-01' BETWEEN
   dbt_valid_from AND dbt_valid_to даёт состояние на дату. */
select * from {{ ref('dim_device_profile') }}

{% endsnapshot %}
