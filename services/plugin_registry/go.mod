module shieldx/services/plugin_registry

go 1.22.0

require (
	shieldx/core/fortress_bridge v0.0.0
	shieldx/sandbox/runner v0.0.0
	github.com/lib/pq v1.10.9
	github.com/tetratelabs/wazero v1.6.0
)

replace shieldx/core/fortress_bridge => ../../core/fortress_bridge
replace shieldx/sandbox/runner => ../../sandbox/runner