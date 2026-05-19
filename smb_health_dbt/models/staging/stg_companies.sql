with source as (
    select * from {{ ref('companies') }}
)

select
    company_id::varchar                       as company_id,
    company_name::varchar                     as company_name,
    industry::varchar                         as industry,
    company_size::varchar                     as company_size,
    state::varchar                            as state,
    region::varchar                           as region,
    signup_date::date                         as signup_date,
    tenure_bucket::varchar                    as tenure_bucket,
    credit_limit::decimal(12, 2)              as credit_limit,
    assigned_csm::varchar                     as assigned_csm,
    health_segment::varchar                   as health_segment,
    dominant_payment_method::varchar          as dominant_payment_method,
    most_common_failure_reason::varchar       as most_common_failure_reason,
    current_timestamp                         as loaded_at
from source
