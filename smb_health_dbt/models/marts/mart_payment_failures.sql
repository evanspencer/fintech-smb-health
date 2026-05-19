with failed_txns as (
    select * from {{ ref('stg_transactions') }}
    where txn_status = 'failed'
),

companies as (
    select company_id, industry, health_segment
    from {{ ref('stg_companies') }}
),

total_failures as (
    select count(*) as total_count
    from failed_txns
),

grouped as (
    select
        t.failure_reason,
        c.industry,
        c.health_segment,
        t.payment_method,
        count(*)                    as failure_count,
        round(sum(t.amount), 2)     as total_amount_failed,
        round(avg(t.amount), 2)     as avg_amount_failed
    from failed_txns t
    join companies c on t.company_id = c.company_id
    group by t.failure_reason, c.industry, c.health_segment, t.payment_method
)

select
    g.failure_reason,
    g.industry,
    g.health_segment,
    g.payment_method,
    g.failure_count,
    g.total_amount_failed,
    g.avg_amount_failed,
    round(g.failure_count::decimal / tf.total_count, 4)    as pct_of_total_failures
from grouped g
cross join total_failures tf
order by g.failure_count desc
