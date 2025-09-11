// Create constraints and indexes for performance
CREATE CONSTRAINT artifact_id_unique IF NOT EXISTS FOR (a:Artifact) REQUIRE a.id IS UNIQUE;
CREATE INDEX artifact_type_idx IF NOT EXISTS FOR (a:Artifact) ON (a.type);
CREATE INDEX artifact_timestamp_idx IF NOT EXISTS FOR (a:Artifact) ON (a.timestamp);
CREATE INDEX artifact_value_idx IF NOT EXISTS FOR (a:Artifact) ON (a.value);

// Create sample data for testing
MERGE (malware:Artifact {id: "mal_001", type: "malware", value: "trojan.exe", timestamp: timestamp()})
SET malware.properties = {risk_score: 0.9, family: "trojan"}

MERGE (ip:Artifact {id: "ip_001", type: "ip", value: "192.168.1.100", timestamp: timestamp()})
SET ip.properties = {country: "US", reputation: "suspicious"}

MERGE (domain:Artifact {id: "dom_001", type: "domain", value: "malicious.com", timestamp: timestamp()})
SET domain.properties = {registrar: "unknown", creation_date: "2024-01-01"}

// Create relationships
MERGE (malware)-[:COMMUNICATES_WITH {timestamp: timestamp()}]->(ip)
MERGE (ip)-[:RESOLVES_TO {timestamp: timestamp()}]->(domain)
MERGE (malware)-[:DOWNLOADS_FROM {timestamp: timestamp()}]->(domain)