package store

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"time"

	_ "github.com/lib/pq"
)

type IOC struct {
	ID              string    `json:"id" db:"id"`
	TenantIDHash    string    `json:"tenant_id_hash" db:"tenant_id_hash"`
	IOCType         string    `json:"ioc_type" db:"ioc_type"`
	ValueHash       string    `json:"value_hash" db:"value_hash"`
	Confidence      float64   `json:"confidence" db:"confidence"`
	TTL             int       `json:"ttl" db:"ttl"`
	FirstSeen       time.Time `json:"first_seen" db:"first_seen"`
	LastSeen        time.Time `json:"last_seen" db:"last_seen"`
	AggregatedCount int       `json:"aggregated_count" db:"aggregated_count"`
}

type Store struct {
	db *sql.DB
}

func NewStore(dbURL string) (*Store, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	store := &Store{db: db}
	if err := store.migrate(); err != nil {
		return nil, fmt.Errorf("migration failed: %w", err)
	}

	return store, nil
}

func (s *Store) migrate() error {
	query := `
	CREATE TABLE IF NOT EXISTS iocs (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		tenant_id_hash VARCHAR(64) NOT NULL,
		ioc_type VARCHAR(50) NOT NULL,
		value_hash VARCHAR(64) NOT NULL,
		confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
		ttl INTEGER NOT NULL,
		first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		aggregated_count INTEGER DEFAULT 1,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);
	
	CREATE INDEX IF NOT EXISTS idx_iocs_tenant_hash ON iocs(tenant_id_hash);
	CREATE INDEX IF NOT EXISTS idx_iocs_value_hash ON iocs(value_hash);
	CREATE INDEX IF NOT EXISTS idx_iocs_type ON iocs(ioc_type);
	CREATE UNIQUE INDEX IF NOT EXISTS idx_iocs_unique ON iocs(tenant_id_hash, value_hash, ioc_type);`

	_, err := s.db.Exec(query)
	return err
}

func (s *Store) SubmitIOC(ioc *IOC) error {
	query := `
	INSERT INTO iocs (tenant_id_hash, ioc_type, value_hash, confidence, ttl)
	VALUES ($1, $2, $3, $4, $5)
	ON CONFLICT (tenant_id_hash, value_hash, ioc_type)
	DO UPDATE SET
		aggregated_count = iocs.aggregated_count + 1,
		last_seen = NOW(),
		confidence = GREATEST(iocs.confidence, EXCLUDED.confidence)
	RETURNING id, first_seen, aggregated_count`

	err := s.db.QueryRow(query, ioc.TenantIDHash, ioc.IOCType, ioc.ValueHash, ioc.Confidence, ioc.TTL).
		Scan(&ioc.ID, &ioc.FirstSeen, &ioc.AggregatedCount)

	return err
}

func (s *Store) QueryIOC(valueHash, iocType string) (*IOC, error) {
	query := `
	SELECT id, tenant_id_hash, ioc_type, value_hash, confidence, ttl, 
		   first_seen, last_seen, aggregated_count
	FROM iocs 
	WHERE value_hash = $1 AND ioc_type = $2 
	AND first_seen + INTERVAL '1 second' * ttl > NOW()`

	ioc := &IOC{}
	err := s.db.QueryRow(query, valueHash, iocType).Scan(
		&ioc.ID, &ioc.TenantIDHash, &ioc.IOCType, &ioc.ValueHash,
		&ioc.Confidence, &ioc.TTL, &ioc.FirstSeen, &ioc.LastSeen,
		&ioc.AggregatedCount,
	)

	if err == sql.ErrNoRows {
		return nil, nil
	}

	return ioc, err
}

func (s *Store) Close() error {
	return s.db.Close()
}

func HashValue(value string) string {
	h := sha256.Sum256([]byte(value))
	return hex.EncodeToString(h[:])
}

func HashTenant(tenantID string) string {
	h := sha256.Sum256([]byte("tenant_salt_2024:" + tenantID))
	return hex.EncodeToString(h[:])
}