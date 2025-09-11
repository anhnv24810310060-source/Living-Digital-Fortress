package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type GraphWriter struct {
	driver neo4j.DriverWithContext
}

type Artifact struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Value      string                 `json:"value"`
	Properties map[string]interface{} `json:"properties"`
	Timestamp  time.Time              `json:"timestamp"`
}

type Relationship struct {
	From       string                 `json:"from"`
	To         string                 `json:"to"`
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	Timestamp  time.Time              `json:"timestamp"`
}

func NewGraphWriter(uri, username, password string) (*GraphWriter, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
	}

	ctx := neo4j.WithBookmarks(neo4j.NewBookmarks())
	err = driver.VerifyConnectivity(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to verify Neo4j connectivity: %w", err)
	}

	return &GraphWriter{driver: driver}, nil
}

func (gw *GraphWriter) CreateNode(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var artifact Artifact
	if err := json.NewDecoder(r.Body).Decode(&artifact); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if artifact.ID == "" || artifact.Type == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	artifact.Timestamp = time.Now()

	ctx := neo4j.WithBookmarks(neo4j.NewBookmarks())
	session := gw.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	cypher := `
	MERGE (n:Artifact {id: $id})
	SET n.type = $type,
		n.value = $value,
		n.timestamp = $timestamp,
		n.properties = $properties
	RETURN n.id as id`

	result, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		return tx.Run(ctx, cypher, map[string]interface{}{
			"id":         artifact.ID,
			"type":       artifact.Type,
			"value":      artifact.Value,
			"timestamp":  artifact.Timestamp.Unix(),
			"properties": artifact.Properties,
		})
	})

	if err != nil {
		log.Printf("Failed to create node: %v", err)
		http.Error(w, "Failed to create node", http.StatusInternalServerError)
		return
	}

	records := result.(*neo4j.Result)
	if records.Next(ctx) {
		nodeID := records.Record().Values[0].(string)
		response := map[string]interface{}{
			"success": true,
			"node_id": nodeID,
			"message": "Node created successfully",
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Failed to encode response: %v", err)
		}
	}
}

func (gw *GraphWriter) CreateEdge(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var rel Relationship
	if err := json.NewDecoder(r.Body).Decode(&rel); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if rel.From == "" || rel.To == "" || rel.Type == "" {
		http.Error(w, "Missing required fields", http.StatusBadRequest)
		return
	}

	rel.Timestamp = time.Now()

	ctx := neo4j.WithBookmarks(neo4j.NewBookmarks())
	session := gw.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	cypher := fmt.Sprintf(`
	MATCH (from:Artifact {id: $from})
	MATCH (to:Artifact {id: $to})
	MERGE (from)-[r:%s]->(to)
	SET r.timestamp = $timestamp,
		r.properties = $properties
	RETURN r`, rel.Type)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		return tx.Run(ctx, cypher, map[string]interface{}{
			"from":       rel.From,
			"to":         rel.To,
			"timestamp":  rel.Timestamp.Unix(),
			"properties": rel.Properties,
		})
	})

	if err != nil {
		log.Printf("Failed to create edge: %v", err)
		http.Error(w, "Failed to create edge", http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"success": true,
		"message": "Edge created successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (gw *GraphWriter) QueryGraph(w http.ResponseWriter, r *http.Request) {
	cypherQuery := r.URL.Query().Get("cypher")
	if cypherQuery == "" {
		http.Error(w, "Missing cypher query parameter", http.StatusBadRequest)
		return
	}

	ctx := neo4j.WithBookmarks(neo4j.NewBookmarks())
	session := gw.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		return tx.Run(ctx, cypherQuery, nil)
	})

	if err != nil {
		log.Printf("Failed to execute query: %v", err)
		http.Error(w, "Query execution failed", http.StatusInternalServerError)
		return
	}

	records := result.(*neo4j.Result)
	var results []map[string]interface{}

	for records.Next(ctx) {
		record := records.Record()
		recordMap := make(map[string]interface{})
		for i, key := range record.Keys {
			recordMap[key] = record.Values[i]
		}
		results = append(results, recordMap)
	}

	response := map[string]interface{}{
		"success": true,
		"results": results,
		"count":   len(results),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (gw *GraphWriter) Close() error {
	ctx := neo4j.WithBookmarks(neo4j.NewBookmarks())
	return gw.driver.Close(ctx)
}

func main() {
	neo4jURI := getEnv("NEO4J_URI", "bolt://localhost:7687")
	neo4jUser := getEnv("NEO4J_USER", "neo4j")
	neo4jPassword := getEnv("NEO4J_PASSWORD", "password")
	port := getEnv("PORT", "8081")

	writer, err := NewGraphWriter(neo4jURI, neo4jUser, neo4jPassword)
	if err != nil {
		log.Fatalf("Failed to initialize graph writer: %v", err)
	}
	defer writer.Close()

	http.HandleFunc("/graph/node", writer.CreateNode)
	http.HandleFunc("/graph/edge", writer.CreateEdge)
	http.HandleFunc("/graph/query", writer.QueryGraph)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy","service":"threatgraph"}`))
	})

	log.Printf("Threat Graph service starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}