with companies as (
    select * from {{ ref('stg_companies') }}
),

payment_behavior as (
    select * from {{ ref('int_company_payment_behavior') }}
),

txn_behavior as (
    select * from {{ ref('int_company_transaction_behavior') }}
)

select
    -- identity + categorical features
    c.company_id,
    c.company_name,
    c.industry,
    c.company_size,
    c.state,
    c.region,
    c.signup_date,
    c.tenure_bucket,
    c.credit_limit,
    c.assigned_csm,
    c.dominant_payment_method,
    c.most_common_failure_reason,

    -- label (pass-through — do not derive)
    c.health_segment,
    case c.health_segment
        when 'healthy'  then 0
        when 'watch'    then 1
        when 'at_risk'  then 2
    end::integer                            as health_segment_encoded,

    -- payment behavior signals
    pb.total_payments,
    pb.missed_payments,
    pb.late_payments,
    pb.missed_payment_rate,
    pb.late_payment_rate,
    pb.avg_days_late,
    pb.max_days_late,
    pb.avg_retry_count,
    pb.payments_last_30d,
    pb.payments_last_90d,

    -- transaction behavior signals
    tb.total_txns,
    tb.failed_txns,
    tb.completed_txns,
    tb.txn_failure_rate,
    tb.total_spend,
    tb.avg_txn_amount,
    tb.spend_last_30d,
    tb.spend_last_90d,
    tb.spend_last_180d,
    tb.spend_trend,
    tb.unique_merchants,
    tb.unique_mcc_categories,
    tb.dominant_mcc_category,

    -- derived features
    round(tb.total_spend / nullif(c.credit_limit, 0), 4)   as credit_utilization,
    current_date                                            as as_of_date

from companies c
left join payment_behavior pb   on c.company_id = pb.company_id
left join txn_behavior tb       on c.company_id = tb.company_id
