

-- Step 1: Find each customer's first purchase month (cohort)
WITH customer_cohorts AS (
    SELECT
        customer_unique_id,
        strftime('%Y-%m', MIN(order_purchase_timestamp)) AS cohort_month,
        COUNT(order_id)                                      AS total_orders,
        ROUND(SUM(total_payment), 2)                         AS total_spent
    FROM master_orders
    GROUP BY customer_unique_id
),

-- Step 2: Summarise by cohort month
cohort_summary AS (
    SELECT
        cohort_month,
        COUNT(customer_unique_id)       AS cohort_size,
        ROUND(SUM(total_spent), 2)     AS cohort_revenue,
        ROUND(AVG(total_spent), 2)     AS avg_spend_per_customer,
        SUM(CASE WHEN total_orders > 1
                 THEN 1 ELSE 0 END)   AS repeat_customers
    FROM customer_cohorts
    GROUP BY cohort_month
)

SELECT
    cohort_month,
    cohort_size,
    cohort_revenue,
    avg_spend_per_customer,
    repeat_customers,
    ROUND(repeat_customers * 100.0 / cohort_size, 1) AS repeat_rate_pct
FROM cohort_summary
ORDER BY cohort_month;