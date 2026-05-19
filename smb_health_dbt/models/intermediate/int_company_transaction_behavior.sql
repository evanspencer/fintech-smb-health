with companies as (
    select company_id
    from {{ ref('stg_companies') }}
),

transactions as (
    select * from {{ ref('stg_transactions') }}
),

aggregated as (
    select
        company_id,
        count(*)                                                                as total_txns,
        sum(is_failed::integer)                                                 as failed_txns,
        sum((txn_status = 'completed')::integer)                                as completed_txns,
        round(sum(is_failed::integer)::decimal / count(*), 4)                   as txn_failure_rate,
        round(sum(amount), 2)                                                   as total_spend,
        round(avg(amount), 2)                                                   as avg_txn_amount,
        round(sum(case when txn_date >= current_date - interval '30 days'
                       then amount else 0 end), 2)                              as spend_last_30d,
        round(sum(case when txn_date >= current_date - interval '90 days'
                       then amount else 0 end), 2)                              as spend_last_90d,
        round(sum(case when txn_date >= current_date - interval '180 days'
                       then amount else 0 end), 2)                              as spend_last_180d,
        count(distinct merchant_name)                                           as unique_merchants,
        count(distinct mcc_category)                                            as unique_mcc_categories
    from transactions
    group by company_id
),

-- Most frequent MCC category on completed transactions only
mcc_counts as (
    select
        company_id,
        mcc_category,
        count(*) as mcc_count
    from transactions
    where txn_status = 'completed'
    group by company_id, mcc_category
),

mcc_ranked as (
    select
        company_id,
        mcc_category,
        row_number() over (
            partition by company_id
            order by mcc_count desc, mcc_category
        ) as rn
    from mcc_counts
)

select
    c.company_id,
    coalesce(a.total_txns, 0)           as total_txns,
    coalesce(a.failed_txns, 0)          as failed_txns,
    coalesce(a.completed_txns, 0)       as completed_txns,
    coalesce(a.txn_failure_rate, 0)     as txn_failure_rate,
    coalesce(a.total_spend, 0)          as total_spend,
    a.avg_txn_amount,
    coalesce(a.spend_last_30d, 0)       as spend_last_30d,
    coalesce(a.spend_last_90d, 0)       as spend_last_90d,
    coalesce(a.spend_last_180d, 0)      as spend_last_180d,
    round(
        (a.spend_last_30d / nullif(a.spend_last_90d / 3.0, 0)) - 1,
        4
    )                                   as spend_trend,
    coalesce(a.unique_merchants, 0)     as unique_merchants,
    coalesce(a.unique_mcc_categories, 0) as unique_mcc_categories,
    mr.mcc_category                     as dominant_mcc_category
from companies c
left join aggregated a      on c.company_id = a.company_id
left join mcc_ranked mr     on c.company_id = mr.company_id and mr.rn = 1
