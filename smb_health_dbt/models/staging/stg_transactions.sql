with source as (
    select * from {{ ref('transactions') }}
)

select
    txn_id::varchar                               as txn_id,
    company_id::varchar                           as company_id,
    amount::decimal(14, 2)                        as amount,
    merchant_name::varchar                        as merchant_name,
    mcc_code::varchar                             as mcc_code,
    mcc_category::varchar                         as mcc_category,
    payment_method::varchar                       as payment_method,
    txn_date::date                                as txn_date,
    txn_status::varchar                           as txn_status,
    coalesce(failure_reason::varchar, 'none')     as failure_reason,
    (txn_status = 'failed')::boolean              as is_failed,
    current_timestamp                             as loaded_at
from source
