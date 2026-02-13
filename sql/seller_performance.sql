
WITH seller_metrics AS (
    SELECT
        i.seller_id,
        s.seller_state,
        COUNT(DISTINCT m.order_id)          AS total_orders,
        ROUND(SUM(m.total_payment), 2)      AS total_revenue,
        ROUND(AVG(m.review_score), 2)       AS avg_rating,
        ROUND(AVG(m.delivery_days), 1)      AS avg_delivery_days,
        ROUND(AVG(m.is_late) * 100, 1)     AS late_delivery_pct,
        COUNT(DISTINCT i.product_id)        AS unique_products
    FROM master_orders m
    JOIN order_items i ON m.order_id = i.order_id
    JOIN sellers s     ON i.seller_id = s.seller_id
    GROUP BY i.seller_id, s.seller_state
)

SELECT
    seller_id,
    seller_state,
    total_orders,
    total_revenue,
    avg_rating,
    avg_delivery_days,
    late_delivery_pct,
    unique_products,
    RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
    CASE
        WHEN RANK() OVER
             (ORDER BY total_revenue DESC) <= 50  THEN 'Platinum'
        WHEN RANK() OVER
             (ORDER BY total_revenue DESC) <= 200 THEN 'Gold'
        WHEN RANK() OVER
             (ORDER BY total_revenue DESC) <= 500 THEN 'Silver'
        ELSE                                           'Bronze'
    END AS seller_tier
FROM seller_metrics
ORDER BY revenue_rank;