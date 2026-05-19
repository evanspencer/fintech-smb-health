with source as (
    select * from {{ ref('payments') }}
)

select
    payment_id::varchar                               as payment_id,
    company_id::varchar                               as company_id,
    amount::decimal(14, 2)                            as amount,
    payment_method::varchar                           as payment_method,
    due_date::date                                    as due_date,
    paid_date::date                                   as paid_date,
    days_late::integer                                as days_late,
    payment_status::varchar                           as payment_status,
    retry_count::integer                              as retry_count,
    period::varchar                                   as period,
    (payment_status in ('late', 'missed'))::boolean   as is_late,
    (payment_status = 'missed')::boolean              as is_missed,
    current_timestamp                                 as loaded_at
from source
