// Find attack propagation chains
MATCH path = (start:Artifact {type: $start_type})-[*1..$max_depth]->(end:Artifact)
WHERE start.properties.risk_score > $min_risk_score
RETURN path, length(path) as chain_length
ORDER BY chain_length DESC
LIMIT $limit;

// Find artifacts connected to specific IOC
MATCH (ioc:Artifact {id: $ioc_id})-[*1..2]-(connected:Artifact)
RETURN DISTINCT connected.id, connected.type, connected.value;

// Find high-risk propagation nodes
MATCH (n:Artifact)-[r]-(connected:Artifact)
WITH n, count(connected) as connections
WHERE connections > $min_connections
RETURN n.id, n.type, connections
ORDER BY connections DESC;