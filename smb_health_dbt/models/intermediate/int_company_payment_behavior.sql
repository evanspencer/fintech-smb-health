with companies as (
    select company_id
    from {{ ref('stg_companies') }}
),

payments as (
    select * from {{ ref('stg_payments') }}
),

aggregated as (
    select
        company_id,
        count(*)                                                            as total_payments,
        sum(is_missed::integer)                                             as missed_payments,
        sum(is_late::integer)                                               as late_payments,
        round(sum(is_missed::integer)::decimal / count(*), 4)               as missed_payment_rate,
        round(sum(is_late::integer)::decimal / count(*), 4)                 as late_payment_rate,
        round(avg(days_late), 2)                                            as avg_days_late,
        max(days_late)                                                      as max_days_late,
        round(avg(retry_count), 4)                                          as avg_retry_count,
        sum(case when due_date >= current_date - interval '30 days'
                 then 1 else 0 end)                                         as payments_last_30d,
        sum(case when due_date >= current_date - interval '90 days'
                 then 1 else 0 end)                                         as payments_last_90d
    from payments
    group by company_id
)

select
    c.company_id,
    coalesce(a.total_payments, 0)       as total_payments,
    coalesce(a.missed_payments, 0)      as missed_payments,
    coalesce(a.late_payments, 0)        as late_payments,
    coalesce(a.missed_payment_rate, 0)  as missed_payment_rate,
    coalesce(a.late_payment_rate, 0)    as late_payment_rate,
    a.avg_days_late,
    a.max_days_late,
    a.avg_retry_count,
    coalesce(a.payments_last_30d, 0)    as payments_last_30d,
    coalesce(a.payments_last_90d, 0)    as payments_last_90d
from companies c
left join aggregated a on c.company_id = a.company_id
