#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/ptrace.h>
#include <linux/sched.h>

struct syscall_event {
    __u32 event_type;    // 1=syscall, 2=network, 3=file
    __u64 timestamp;
    __u32 pid;
    __u32 syscall_nr;
    __u64 args[6];
    __s64 ret_code;
};

struct network_event {
    __u32 event_type;    // 2
    __u64 timestamp;
    __u32 pid;
    __u16 src_port;
    __u16 dst_port;
    __u32 src_ip;
    __u32 dst_ip;
    __u8 protocol;
    __u32 bytes;
};

struct file_event {
    __u32 event_type;    // 3
    __u64 timestamp;
    __u32 pid;
    __u32 mode;
    __u8 operation;      // 0=read, 1=write
    __u8 success;
    char path[256];
};

struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} events SEC(".maps");

// Syscall filtering map - only monitor dangerous syscalls
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u8));
} dangerous_syscalls SEC(".maps");

// Process filtering - only monitor sandbox processes
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u8));
} monitored_pids SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_openat")
int trace_syscall_enter(struct trace_event_raw_sys_enter* ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    // Check if this PID should be monitored
    __u8 *monitored = bpf_map_lookup_elem(&monitored_pids, &pid);
    if (!monitored) {
        return 0;
    }
    
    __u32 syscall_nr = ctx->id;
    
    // Check if this is a dangerous syscall
    __u8 *dangerous = bpf_map_lookup_elem(&dangerous_syscalls, &syscall_nr);
    if (!dangerous) {
        return 0;
    }
    
    struct syscall_event event = {};
    event.event_type = 1;
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.syscall_nr = syscall_nr;
    
    // Safely copy syscall arguments
    for (int i = 0; i < 6; i++) {
        if (i < sizeof(ctx->args) / sizeof(ctx->args[0])) {
            event.args[i] = ctx->args[i];
        }
    }
    
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    return 0;
}

SEC("tracepoint/syscalls/sys_exit_openat")
int trace_syscall_exit(struct trace_event_raw_sys_exit* ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    __u8 *monitored = bpf_map_lookup_elem(&monitored_pids, &pid);
    if (!monitored) {
        return 0;
    }
    
    __u32 syscall_nr = ctx->id;
    __u8 *dangerous = bpf_map_lookup_elem(&dangerous_syscalls, &syscall_nr);
    if (!dangerous) {
        return 0;
    }
    
    struct syscall_event event = {};
    event.event_type = 1;
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.syscall_nr = syscall_nr;
    event.ret_code = ctx->ret;
    
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    return 0;
}

SEC("tracepoint/syscalls/sys_enter_socket")
int trace_network(struct trace_event_raw_sys_enter* ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    __u8 *monitored = bpf_map_lookup_elem(&monitored_pids, &pid);
    if (!monitored) {
        return 0;
    }
    
    struct network_event event = {};
    event.event_type = 2;
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    
    // Extract socket family and type from args
    if (ctx->args[0] == AF_INET || ctx->args[0] == AF_INET6) {
        event.protocol = (ctx->args[1] == SOCK_STREAM) ? 6 : 17; // TCP : UDP
        
        bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    }
    
    return 0;
}

SEC("tracepoint/syscalls/sys_enter_write")
int trace_file_write(struct trace_event_raw_sys_enter* ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    __u8 *monitored = bpf_map_lookup_elem(&monitored_pids, &pid);
    if (!monitored) {
        return 0;
    }
    
    struct file_event event = {};
    event.event_type = 3;
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.operation = 1; // write
    
    // Get file descriptor and try to resolve path
    int fd = (int)ctx->args[0];
    
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    return 0;
}

SEC("tracepoint/syscalls/sys_enter_read")
int trace_file_read(struct trace_event_raw_sys_enter* ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    __u8 *monitored = bpf_map_lookup_elem(&monitored_pids, &pid);
    if (!monitored) {
        return 0;
    }
    
    struct file_event event = {};
    event.event_type = 3;
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.operation = 0; // read
    
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    return 0;
}

// Memory protection changes (mprotect) - critical for exploit detection
SEC("tracepoint/syscalls/sys_enter_mprotect")
int trace_mprotect(struct trace_event_raw_sys_enter* ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    __u8 *monitored = bpf_map_lookup_elem(&monitored_pids, &pid);
    if (!monitored) {
        return 0;
    }
    
    struct syscall_event event = {};
    event.event_type = 1;
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.syscall_nr = __NR_mprotect;
    
    // mprotect args: addr, len, prot
    event.args[0] = ctx->args[0]; // address
    event.args[1] = ctx->args[1]; // length
    event.args[2] = ctx->args[2]; // protection flags
    
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    return 0;
}

// Process creation monitoring
SEC("tracepoint/syscalls/sys_enter_clone")
int trace_clone(struct trace_event_raw_sys_enter* ctx) {
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    __u8 *monitored = bpf_map_lookup_elem(&monitored_pids, &pid);
    if (!monitored) {
        return 0;
    }
    
    struct syscall_event event = {};
    event.event_type = 1;
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid;
    event.syscall_nr = __NR_clone;
    
    // Clone flags indicate type of process/thread creation
    event.args[0] = ctx->args[0]; // clone_flags
    event.args[1] = ctx->args[1]; // newsp
    
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    return 0;
}

char _license[] SEC("license") = "GPL";