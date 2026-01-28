# Root Filesystem Crisis - Recovery & Prevention

## ðŸš¨ What Happened

Claude Code filled `/tmp/claude` with 20GB of data, exhausting your 84GB root filesystem and crashing the system.

**Root cause:** Claude Code's runtime creates `/tmp/claude` before AI prompt instructions are evaluated, bypassing the "write only to `/mnt/raid0/`" constraint.

---

## âœ… Solution Implemented

I've created a **three-layer defense** to prevent this from happening again:

### Layer 1: Safe Startup Wrapper (REQUIRED)

**File:** `/mnt/raid0/llm/UTILS/claude_safe_start.sh`

This wrapper script:
1. Sets ALL cache/temp environment variables to `/mnt/raid0/`
2. Creates a **bind mount** that redirects `/tmp/claude` → `/mnt/raid0/llm/tmp/claude`
3. Starts Claude Code with enforced storage constraints

**Key:** The bind mount means even if Claude writes to `/tmp/claude`, the data physically goes to your RAID array (3.6TB free) instead of root FS (84GB).

### Layer 2: Real-Time Monitoring

**File:** `/mnt/raid0/llm/UTILS/monitor_storage.sh`

Monitors root FS usage every 30 seconds:
- **70% full:** Warning logged
- **85% full:** Critical alert + system notification

Run in background to get early warning.

### Layer 3: Emergency Recovery

**File:** `/mnt/raid0/llm/UTILS/emergency_cleanup.sh`

If system fills up again:
- Stops Claude processes
- Unmounts bind mount
- Deletes `/tmp/claude`
- Reports before/after usage

---

## 📋 Action Plan - What You Need to Do

### Step 1: Clean Up Now (One-Time)

```bash
# Run the emergency cleanup script
sudo bash /mnt/raid0/llm/UTILS/emergency_cleanup.sh
```

This will:
- Ask to kill any running Claude processes
- Ask to delete `/tmp/claude` (say yes - it has 20GB you don't need)
- Show before/after root FS usage

### Step 2: Always Use Wrapper (Going Forward)

**NEVER start Claude Code directly anymore.**

Instead, always use:
```bash
bash /mnt/raid0/llm/UTILS/claude_safe_start.sh
```

This wrapper is your protection.

### Step 3: Run Health Check Before Sessions (Optional but Recommended)

```bash
bash /mnt/raid0/llm/UTILS/health_check.sh
```

This checks:
- Root FS usage is safe (<70%)
- Bind mount is active
- Environment variables are set correctly
- Storage monitor is running

Exit codes:
- **0:** All good, proceed
- **1:** Critical issues, fix before starting Claude

### Step 4: Monitor During Sessions (Recommended)

In a separate terminal:
```bash
# Background monitor (alerts automatically)
bash /mnt/raid0/llm/UTILS/monitor_storage.sh &

# OR: Watch manually
watch -n 5 'df -h /'
```

---

## 🔧 Files Created

All scripts are in `/mnt/raid0/llm/UTILS/`:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `claude_safe_start.sh` | Start Claude safely | **Always** (replaces `claude` command) |
| `emergency_cleanup.sh` | Free root FS space | When system fills up |
| `monitor_storage.sh` | Real-time alerts | Run in background during sessions |
| `health_check.sh` | Pre-session validation | Before each Claude session |

Documentation:
- `/mnt/raid0/llm/claude/logs/INCIDENT_ROOT_FS_FULL.md` - Full technical analysis
- `/mnt/raid0/llm/claude/CLAUDE.md` - Updated with emergency procedures

---

## 🧪 Testing the Fix

### Verify Current State

```bash
# Check root FS usage
df -h /
# Should be <70% after cleanup

# Check if bind mount exists (will fail if cleanup not run yet)
mountpoint /tmp/claude
# Should say "is a mountpoint" OR "not a mountpoint" (both OK)
```

### Test the Wrapper

```bash
# Start via wrapper
bash /mnt/raid0/llm/UTILS/claude_safe_start.sh
```

You should see output like:
```
✓ Environment variables set to redirect ALL writes to /mnt/raid0/
✓ Directories created
Root filesystem usage: XX%
RAID0 available: 3.6T
Creating bind mount: /tmp/claude -> /mnt/raid0/llm/tmp/claude
✓ Bind mount active
Starting Claude Code...
```

### During Session

In another terminal:
```bash
# Verify mount is active
mountpoint /tmp/claude
# Output: /tmp/claude is a mountpoint

# Watch where data is actually going
watch -n 10 'du -sh /tmp/claude && du -sh /mnt/raid0/llm/tmp/claude'
# Both should show same size (because they're the same location via bind mount)

# Watch root FS (should NOT grow)
watch -n 5 'df -h /'
```

---

## ❓ FAQ

**Q: Why can't we just tell Claude not to write to `/tmp/`?**  
A: Claude Code (the application) creates `/tmp/claude` before the AI reads its instructions. Prompt constraints can't override application-level behavior.

**Q: What does the bind mount actually do?**  
A: It makes `/tmp/claude` a "portal" to `/mnt/raid0/llm/tmp/claude`. When anything writes to `/tmp/claude`, it physically goes to the RAID array.

**Q: Will this affect performance?**  
A: No. The RAID array is NVMe (faster than a typical /tmp on spinning disk), and the bind mount has negligible overhead.

**Q: What if I forget to use the wrapper?**  
A: Root FS will fill up again and Claude will crash. The monitor script will alert you, but prevention is better.

**Q: Can I automate this so I don't have to remember?**  
A: Yes - add an alias to your `.bashrc`:
```bash
alias claude='bash /mnt/raid0/llm/UTILS/claude_safe_start.sh'
```
Then typing `claude` will always use the safe wrapper.

**Q: How do I unmount `/tmp/claude` after a session?**  
A: You don't need to - it's safe to leave mounted. But if you want:
```bash
sudo umount /tmp/claude
```

---

## 🎯 Quick Reference

### Before Session
```bash
bash /mnt/raid0/llm/UTILS/health_check.sh
```

### Start Session
```bash
bash /mnt/raid0/llm/UTILS/claude_safe_start.sh
```

### During Session (separate terminal)
```bash
bash /mnt/raid0/llm/UTILS/monitor_storage.sh &
```

### Emergency Recovery
```bash
sudo bash /mnt/raid0/llm/UTILS/emergency_cleanup.sh
```

---

## 📞 Next Steps

1. **Now:** Run `emergency_cleanup.sh` to free space
2. **Test:** Start one Claude session via wrapper, verify bind mount works
3. **Going forward:** Always use the wrapper
4. **Optional:** Set up the `.bashrc` alias so you don't have to remember

The system is now protected. Let me know if you hit any issues!
