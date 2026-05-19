with transactions as (
    select * from {{ ref('stg_transactions') }}
),

companies as (
    select company_id, industry, health_segment
    from {{ ref('stg_companies') }}
)

select
    t.txn_date,
    c.industry,
    c.health_segment,
    count(*)                                                        as daily_txn_count,
    round(sum(t.amount), 2)                                         as daily_volume,
    sum(t.is_failed::integer)                                       as daily_failed_count,
    round(sum(t.is_failed::integer)::decimal / count(*), 4)         as daily_failure_rate
from transactions t
join companies c on t.company_id = c.company_id
group by t.txn_date, c.industry, c.health_segment
order by t.txn_date
