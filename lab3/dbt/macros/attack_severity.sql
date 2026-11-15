{% macro attack_severity(attack, subtype) -%}
{# Макрос повторяющейся бизнес-логики: тяжесть атаки по подтипу (документированная шкала).
   mirai/gafgyt подтипы из домена датасета; неизвестные → 1 (наименьшая тяжесть). #}
case
  when {{ attack }} = 'Normal' then 0
  when {{ subtype }} in ('DDoS', 'UDPfrag', 'greip') then 3
  when {{ subtype }} in ('TCP', 'UDPSlowRate', 'ARP', 'Scan', 'scan') then 2
  else 1
end
{%- endmacro %}
