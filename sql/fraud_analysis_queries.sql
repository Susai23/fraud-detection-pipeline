-- Fraud rate by region
SELECT region, COUNT(*) AS total_claims,
       SUM(CASE WHEN fraud_reported = 'Y' THEN 1 ELSE 0 END) AS fraud_claims,
       ROUND(100.0 * SUM(CASE WHEN fraud_reported = 'Y' THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM claims GROUP BY region ORDER BY fraud_rate_pct DESC;

-- Fraud rate by incident severity
SELECT incident_severity, COUNT(*) AS total_claims,
       ROUND(100.0 * SUM(CASE WHEN fraud_reported = 'Y' THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM claims GROUP BY incident_severity ORDER BY fraud_rate_pct DESC;

-- Fraud rate: low-crime theft claims vs. other (demonstrates real-time enrichment join)
SELECT
    CASE WHEN incident_type = 'Vehicle Theft' AND total_crimes_nearby < 10 THEN 'Low-crime theft claim'
         ELSE 'Other' END AS claim_pattern,
    COUNT(*) AS total_claims,
    ROUND(100.0 * SUM(CASE WHEN fraud_reported = 'Y' THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM claims WHERE total_crimes_nearby IS NOT NULL
GROUP BY claim_pattern;
