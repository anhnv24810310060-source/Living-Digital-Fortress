#!/bin/bash

GRAPH_SERVICE=${GRAPH_SERVICE:-"http://localhost:8081"}

echo "=== Threat Graph Query Interface ==="
echo "Graph Service: $GRAPH_SERVICE"
echo

# Function to execute cypher query
query_graph() {
    local cypher="$1"
    echo "Executing: $cypher"
    curl -s "$GRAPH_SERVICE/graph/query?cypher=$(echo "$cypher" | sed 's/ /%20/g')" | jq .
    echo
}

# Sample queries
echo "1. Show all artifacts:"
query_graph "MATCH (n:Artifact) RETURN n.id, n.type, n.value LIMIT 10"

echo "2. Show attack chains (relationships):"
query_graph "MATCH (a:Artifact)-[r]->(b:Artifact) RETURN a.id, type(r), b.id LIMIT 10"

echo "3. Find high-risk artifacts:"
query_graph "MATCH (n:Artifact) WHERE n.properties.risk_score > 0.8 RETURN n.id, n.type, n.properties.risk_score"

echo "4. Show propagation paths:"
query_graph "MATCH path = (start:Artifact)-[*1..3]->(end:Artifact) WHERE start.type = 'malware' RETURN path LIMIT 5"

echo "5. Node count by type:"
query_graph "MATCH (n:Artifact) RETURN n.type, count(n) as count ORDER BY count DESC"

echo "=== Interactive Mode ==="
echo "Enter Cypher queries (type 'exit' to quit):"

while true; do
    read -p "cypher> " cypher
    if [ "$cypher" = "exit" ]; then
        break
    fi
    if [ -n "$cypher" ]; then
        query_graph "$cypher"
    fi
done