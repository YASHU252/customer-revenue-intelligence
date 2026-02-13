
WITH rfm_raw AS (
    -- Calculate R, F, M values per customer
    SELECT
        customer_unique_id,
        ROUND(julianday('2018-10-17') -
              julianday(MAX(order_purchase_timestamp)), 0) AS recency_days,
        COUNT(order_id)                                       AS frequency,
        ROUND(SUM(total_payment), 2)                          AS monetary
    FROM master_orders
    GROUP BY customer_unique_id
),

rfm_scored AS (
    -- Score each dimension 1-4 using NTILE
    -- Recency: LOWER days = BETTER = higher score (hence DESC)
    SELECT
        customer_unique_id,
        recency_days,
        frequency,
        monetary,
        NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(4) OVER (ORDER BY frequency   ASC)  AS f_score,
        NTILE(4) OVER (ORDER BY monetary    ASC)  AS m_score
    FROM rfm_raw
)

SELECT
    customer_unique_id,
    recency_days,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    (r_score + f_score + m_score)  AS rfm_total,
    CASE
        WHEN (r_score + f_score + m_score) >= 10 THEN 'Champions'
        WHEN (r_score + f_score + m_score) >= 7  THEN 'Loyal Customers'
        WHEN (r_score + f_score + m_score) >= 5  THEN 'Potential Loyalists'
        WHEN r_score <= 2                         THEN 'At Risk'
        ELSE                                            'Need Attention'
    END AS customer_segment
FROM rfm_scored
ORDER BY rfm_total DESC;