
WITH category_revenue AS (
    SELECT
        product_category_name_english           AS category,
        COUNT(DISTINCT order_id)               AS order_count,
        ROUND(SUM(total_payment), 2)          AS revenue
    FROM master_orders
    WHERE product_category_name_english IS NOT NULL
    GROUP BY category
),

pareto AS (
    SELECT
        category,
        order_count,
        revenue,
        -- Running total revenue (cumulative sum)
        SUM(revenue) OVER (
            ORDER BY revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue,
        -- Total revenue for percentage calc
        SUM(revenue) OVER () AS total_revenue
    FROM category_revenue
)

SELECT
    category,
    order_count,
    revenue,
    ROUND(revenue * 100.0 / total_revenue, 1)            AS revenue_pct,
    ROUND(cumulative_revenue * 100.0 / total_revenue, 1)   AS cumulative_pct,
    CASE WHEN cumulative_revenue * 100.0
                / total_revenue <= 80
         THEN 'Top 80% Revenue'
         ELSE 'Long Tail'
    END AS pareto_group
FROM pareto
ORDER BY revenue DESC;