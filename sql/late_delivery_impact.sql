
WITH delivery_analysis AS (
    SELECT
        order_id,
        review_score,
        delivery_days,
        is_late,
        -- Bin delivery time into ranges
        CASE
            WHEN delivery_days <= 7  THEN '1. Within 1 week'
            WHEN delivery_days <= 14 THEN '2. 1-2 weeks'
            WHEN delivery_days <= 21 THEN '3. 2-3 weeks'
            WHEN delivery_days <= 30 THEN '4. 3-4 weeks'
            ELSE                          '5. Over 1 month'
        END AS delivery_bucket
    FROM master_orders
    WHERE review_score IS NOT NULL
      AND delivery_days IS NOT NULL
)

SELECT
    delivery_bucket,
    COUNT(*)                              AS order_count,
    ROUND(AVG(review_score), 2)          AS avg_review_score,
    ROUND(AVG(delivery_days), 1)         AS avg_delivery_days,
    -- % of orders in this bucket that were late
    ROUND(AVG(is_late) * 100, 1)         AS late_pct,
    -- % with 5-star reviews
    ROUND(SUM(CASE WHEN review_score = 5
                    THEN 1 ELSE 0 END)
          * 100.0 / COUNT(*), 1)         AS five_star_pct
FROM delivery_analysis
GROUP BY delivery_bucket
ORDER BY delivery_bucket;